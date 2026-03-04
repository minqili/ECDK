from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
from .loss import CrossEntropyLabelSmooth
from os.path import exists
import os
import math


def reweight_teacher_probs_corrected(teacher_logits, targets, temperature):
    B, C = teacher_logits.shape
    t = F.softmax(teacher_logits / temperature, dim=1)
    t_y = t.gather(1, targets.unsqueeze(1))
    t_max, t_argmax = t.max(dim=1, keepdim=True)
    teacher_wrong = (t_argmax.squeeze(1) != targets).float().unsqueeze(1)
    alpha_min = (t_max - t_y) / (1 - t_y + 1e-6)
    alpha = torch.where(
        teacher_wrong.bool(),
        alpha_min,
        torch.zeros_like(t_y)
    )
    t_y_new = t_y + alpha * (1 - t_y)
    scale = (1 - t_y_new) / (1 - t_y + 1e-6)
    scale = scale.expand(-1, C)
    t_new = t * scale
    gt_mask = F.one_hot(targets, num_classes=C).bool()
    t_new[gt_mask] = t_y_new.squeeze(1)
    return t_new


class SoftTarget_none(nn.Module):
    def __init__(self, T):
        super(SoftTarget_none, self).__init__()
        self.T = T

    def forward(self, out_s, out_t, is_prob=False):
        student_log_prob = F.log_softmax(out_s / self.T, dim=1)
        if is_prob:
            teacher_prob = out_t
        else:
            teacher_prob = F.softmax(out_t / self.T, dim=1)

        loss = F.kl_div(student_log_prob, teacher_prob, reduction='none') * (self.T ** 2)
        return loss.sum(dim=1)


class ECKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ECKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.temperatures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.lambda_mut_base = getattr(cfg.KD.LOSS, "MUTUAL_WEIGHT", 0.6)
        self.criterionKD_none = SoftTarget_none

        # warmup 超参
        self.mut_start_epoch = getattr(cfg.KD.LOSS, "MUTUAL_START_EPOCH", 20)
        self.mut_rampup_epochs = getattr(cfg.KD.LOSS, "MUTUAL_RAMPUP_EPOCHS", 10)

    def _mutual_loss_per_sample(self, logits_a, logits_b, T_mut):
        log_p_a = F.log_softmax(logits_a / T_mut, dim=1)
        p_b = F.softmax(logits_b / T_mut, dim=1)
        kl_ab_per = F.kl_div(log_p_a, p_b, reduction='none').sum(dim=1) * (T_mut ** 2)

        log_p_b = F.log_softmax(logits_b / T_mut, dim=1)
        p_a = F.softmax(logits_a / T_mut, dim=1)
        kl_ba_per = F.kl_div(log_p_b, p_a, reduction='none').sum(dim=1) * (T_mut ** 2)

        return 0.5 * (kl_ab_per + kl_ba_per)  # [N]

    def _get_lambda_mut(self, epoch):
        if epoch < self.mut_start_epoch:
            return 0.0
        rampup_progress = min(1.0, (epoch - self.mut_start_epoch) / self.mut_rampup_epochs)
        return self.lambda_mut_base * rampup_progress

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        epoch = kwargs.get("epoch", 0)
        lambda_mut = self._get_lambda_mut(epoch)

        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)

        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        device = logits_student_weak.device
        batch_size = target.size(0)

        pred1 = logits_teacher_weak.argmax(dim=1)
        pred2 = logits_teacher_strong.argmax(dim=1)
        correct1 = pred1.eq(target)
        correct2 = pred2.eq(target)

        both_correct_mask = correct1 & correct2
        only_t1_correct_mask = correct1 & ~correct2
        only_t2_correct_mask = ~correct1 & correct2

        loss_kd = torch.zeros(batch_size, device=device)

        for T in self.temperatures:
            criterionKD = self.criterionKD_none(T)

            if both_correct_mask.any():
                loss_kd[both_correct_mask] += (
                        criterionKD(logits_student_weak[both_correct_mask],
                                    logits_teacher_weak[both_correct_mask]) +
                        criterionKD(logits_student_strong[both_correct_mask],
                                    logits_teacher_strong[both_correct_mask])
                )

            if only_t1_correct_mask.any():
                corrected_logits_strong = reweight_teacher_probs_corrected(
                    logits_teacher_strong[only_t1_correct_mask],
                    target[only_t1_correct_mask],
                    T
                )
                loss_kd[only_t1_correct_mask] += (
                        criterionKD(logits_student_weak[only_t1_correct_mask],
                                    logits_teacher_weak[only_t1_correct_mask]) +
                        criterionKD(logits_student_strong[only_t1_correct_mask],
                                    corrected_logits_strong,
                                    is_prob=True)
                )

            if only_t2_correct_mask.any():
                corrected_logits_weak = reweight_teacher_probs_corrected(
                    logits_teacher_weak[only_t2_correct_mask],
                    target[only_t2_correct_mask],
                    T
                )
                loss_kd[only_t2_correct_mask] += (
                        criterionKD(logits_student_strong[only_t2_correct_mask],
                                    logits_teacher_strong[only_t2_correct_mask]) +
                        criterionKD(logits_student_weak[only_t2_correct_mask],
                                    corrected_logits_weak,
                                    is_prob=True)
                )

            if lambda_mut > 0:
                mut_per_sample = self._mutual_loss_per_sample(
                    logits_student_weak, logits_student_strong, target, T
                )
                loss_kd += lambda_mut * mut_per_sample

        loss_kd = loss_kd.sum() / (batch_size * len(self.temperatures))

        loss_ce = F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target)

        losses_dict = {
            "loss_ce": self.ce_loss_weight * loss_ce,
            "loss_kd": self.kd_loss_weight * loss_kd,
        }

        return logits_student_weak, losses_dict
