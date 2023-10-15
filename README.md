# Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification (NeurIPS, 2023)

by Jintong Gao<sup>1</sup>, He Zhao<sup>2</sup>, Zhuo Li<sup>3,4</sup>, Dandan Guo<sup>1</sup>

<sup>1</sup>Jilin University, <sup>2</sup>CSIRO's Data61, <sup>3</sup>Shenzhen Research Institute of Big Data, <sup>4</sup>The Chinese University of Hong Kong, Shenzhen

This is the official implementation of [Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification](https://openreview.net/forum?id=M7FQpIdo0X&noteId=a0mlRwgug6) in PyTorch.

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

### CIFAR-LT

CIFAR-10-LT (ERM-DRW + OTmix):

```
python cifar_train.py --dataset cifar10 --num_classes 10 --loss_type ERM --train_rule DRW --data_aug OT --gpu 0
```

CIFAR-100-LT (BALMS + OTmix):

```
python train.py --dataset cifar100 --num_classes 100 --loss_type BALMS --train_rule None --data_aug OT
```
### ImageNet-LT

ERM + OTmix:

```
python ImageNet_train.py --dataset Imagenet-LT --num_classes 1000 --loss_type ERM --train_rule None --data_aug OT
```

### iNaturalist 2018

DRW + OTmix:

```
python iNaturalist 2018_train.py --dataset Imagenet-LT --num_classes 1000 --loss_type ERM --train_rule None --data_aug OT
```

## Evaluation

To evaluate my model, run:

CIFAR-LT
```
python test.py --root path --dataset cifar10 --arch resnet32 --num_classes 10 --gpu 0 --resume model_path
```

ImageNet-LT

```
python test.py --root path --dataset Imagenet-LT --arch resnet50 --num_classes 1000 --resume model_path
```

iNaturalist 2018

```
python test.py --root path --dataset iNat18 --arch resnet50 --num_classes 8142 --resume model_path
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
