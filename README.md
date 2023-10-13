# Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification (NeurIPS, 2023)

by Jintong Gao<sup>1</sup>, He Zhao<sup>2</sup>, Zhuo Li<sup>3,4</sup>, Dandan Guo<sup>1</sup>

<sup>1</sup>Jilin University, <sup>2</sup>CSIRO's Data61, <sup>3</sup>Shenzhen Research Institute of Big Data, <sup>4</sup>The Chinese University of Hong Kong, Shenzhen

This is the official implementation of [Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification](https://pages.github.com/](https://openreview.net/forum?id=M7FQpIdo0X&noteId=a0mlRwgug6) in PyTorch.

## Requirements:

All codes are written by Python 3.6 with 

```
PyTorch >=1.5
torchvision >=0.6
TensorboardX 1.9
Numpy 1.17.3
POT 0.9.0
```

## Training

To train the model(s) in the paper, run this command:

CIFAR-10:

```
python train.py --dataset cifar10 --num_classes 10 --loss_type ERM --train_rule None --data_aug OT
```

CIFAR-100:

```
python train.py --dataset cifar100 --num_classes 100 --loss_type ERM --train_rule DRW --data_aug OT
```

## Evaluation

To evaluate my model, run:

```
python test.py --resume path
```

## Citation

If you find our paper and repo useful, please cite our paper.

```
@inproceedings{
anonymous2023enhancing,
title={Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=M7FQpIdo0X}
}
```
