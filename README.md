## Dual-Perspective Error Correction for Enhanced Knowledge Distillation in Deep Learning


### Abstract

Deep neural networks have achieved remarkable success across various computer vision tasks, yet their deployment in resource-constrained environments remains challenging due to computational and memory costs. Knowledge distillation (KD) offers a promising solution by transferring knowledge from a large teacher model to a compact student model. However, teacher models, despite their high accuracy, can produce biased predictions, leading to misleading supervision. To address this, we introduce Error-Correcting Knowledge Distillation (ECKD), a framework that integrates probability calibration and a dual-view data selection strategy. Probability calibration adjusts the teacher's predictive distribution using ground-truth labels to mitigate misleading supervision, while the dual-view data selection strategy filters out samples with large prediction bias. Extensive experiments on CIFAR-100, Tiny-ImageNet, and ImageNet-1K demonstrate that ECKD achieves highly competitive performance, consistently matching or exceeding the accuracy of prior KD methods.

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### ECKD Framework

<div style="text-align:center"><img src="img/ECKD.png" width="100%" ></div>

### CIFAR-100


- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  python3 tools/train_ours.py --cfg configs/cifar100/eckd/res32x4_res8x4.yaml 
  ```

### Training on Tiny-ImageNet

- Download the dataset at <http://cs231n.stanford.edu/tiny-imagenet-200.zip> and put them to `./data/tiny-imagenet-200`

  ```bash
  python3 tools/train_ours.py --cfg configs/tinyimagenet200/eckd/r34_r18.yaml
  ```

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  python3 tools/train_ours.py --cfg configs/imagenet/r34_r18/eckd.yaml
  ```

# Weight
The weights of student models are available at [Baidu](https://pan.baidu.com/s/1Y-6bKb8iZg8JC80L3QMIOw?pwd=e82i) or [Google](https://drive.google.com/file/d/1oD30nuL03w1eCFVCcWMHByYeEotyHBaU/view?usp=sharing).

# Acknowledgement
Thanks for the contributions to the codebase. The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>) and [mlkd](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>).


# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX

@article{eckd2026,
  title={Dual-Perspective Error Correction for Enhanced Knowledge Distillation in Deep Learning},
  author={Yu Wang, Minqi Li, Kaibing Zhang, Xiangjian He, Xiaomin Ma},
  journal={},
  volume={},
  number={},
  pages={},
  year={2026},
  publisher={}
}

```
