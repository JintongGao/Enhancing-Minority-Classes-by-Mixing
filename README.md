# Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification (NeurIPS, 2023)

by Jintong Gao<sup>1</sup>, He Zhao<sup>2</sup>, Zhuo Li<sup>3,4</sup>, Dandan Guo<sup>1</sup>

<sup>1</sup>Jilin University, <sup>2</sup>CSIRO's Data61, <sup>3</sup>Shenzhen Research Institute of Big Data, <sup>4</sup>The Chinese University of Hong Kong, Shenzhen

This is the official implementation of "Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification" in PyTorch.

## Requirements:

All codes are written by Python 3.6 with PyTorch >=1.5  torchvision >=0.6  TensorboardX 1.9  Numpy 1.17.3  POT 0.9.0

## Training

To train the model(s) in the paper, run this command:

CIFAR-10:

```
python train.py --dataset cifar10 --num_classes 10 --loss_type ERM --train_rule None --data_aug OT
```

CIFAR-100:
python train.py --dataset cifar100 --num_classes 100 --loss_type ERM --train_rule DRW --data_aug OT

## Evaluation

To evaluate my model, run:

python test.py --resume path 

Abstract: Real-world data usually confronts severe class-imbalance problems, where several majority classes have a significantly larger presence in the training set than minority classes. One effective solution is using mixup-based methods to generate synthetic samples to enhance the presence of minority classes. Previous approaches mix the background images from the majority classes and foreground images from the minority classes in a random manner, which ignores the sample-level semantic similarity, possibly resulting in less reasonable or less useful images. In this work, we propose an adaptive image-mixing method based on optimal transport (OT) to incorporate both class-level and sample-level information, which is able to generate semantically reasonable and meaningful mixed images for minority classes. Due to its flexibility, our method can be combined with existing long-tailed classification methods to enhance their performance and it can also serve as a general data augmentation method for balanced datasets. Extensive experiments indicate that our method achieves effective performance for long-tailed classification tasks.

## Citation

If you find our paper and repo useful, please cite our paper.

@inproceedings{
anonymous2023enhancing,
title={Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=M7FQpIdo0X}
}
