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
python cifar_train.py --dataset cifar100 --num_classes 100 --loss_type BALMS --train_rule None --data_aug OT --gpu 0
```
### ImageNet-LT

ERM + OTmix:

```
python imagenet_train.py --root path --dataset Imagenet-LT --num_classes 1000 --loss_type ERM --train_rule None --epochs 200 --data_aug OT
```

### iNaturalist 2018

DRW + OTmix:

```
python iNat18_train.py--root path --dataset iNat18 --num_classes 8142 --loss_type ERM --train_rule DRW --epochs 210 --data_aug OT
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

## Pretrained models

CIFAR-LT [Google drive](https://drive.google.com/drive/folders/1gXtHw-LHDOzywzsyVzYny6ghwK95n_gT/)

ImageNet-LT [Google drive](https://drive.google.com/drive/folders/11WfAI0Epo3Bus37hTeAwBCyhSUzjHEA_)

iNaturalist 2018 [Google drive](https://drive.google.com/drive/folders/1AarCBLI8JHaLGDMGZnvEBPBnmiIogiwD/)

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
@inproceedings{DBLP:conf/nips/GaoZLG23,
  author       = {Jintong Gao and He Zhao and Zhuo Li and Dandan Guo},
  title        = {Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification},
  booktitle    = {Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2023}
}
```

## Concat

If you have any questions when running the code, please feel free to concat us by emailing

+ Jintong Gao ([gaojt20@mails.jlu.edu.cn](mailto:gaojt20.mails.jlu.edu.cn))
