# Enhancing-Minority-Classes-by-Mixing

Code for "Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification", in NeurIPS 2023.

Requirements:

Python 3.6  PyTorch >=1.5  torchvision >=0.6  TensorboardX 1.9  Numpy 1.17.3  POT 0.9.0

## Training

To train the model(s) in the paper, run this command:

CIFAR-10:
python train.py --root data --dataset cifar10 --num_classes 10 --loss_type ERM --train_rule None

CIFAR-100:
python train.py --root data --dataset cifar100 --num_classes 100 --loss_type ERM --train_rule None

## Evaluation

To evaluate my model, run:

python test.py --resume path 

Abstract: Real-world data usually confronts severe class-imbalance problems, where several majority classes have a significantly larger presence in the training set than minority classes. One effective solution is using mixup-based methods to generate synthetic samples to enhance the presence of minority classes. Previous approaches mix the background images from the majority classes and foreground images from the minority classes in a random manner, which ignores the sample-level semantic similarity, possibly resulting in less reasonable or less useful images. In this work, we propose an adaptive image-mixing method based on optimal transport (OT) to incorporate both class-level and sample-level information, which is able to generate semantically reasonable and meaningful mixed images for minority classes. Due to its flexibility, our method can be combined with existing long-tailed classification methods to enhance their performance and it can also serve as a general data augmentation method for balanced datasets. Extensive experiments indicate that our method achieves effective performance for long-tailed classification tasks.

@inproceedings{
anonymous2023enhancing,
title={Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=M7FQpIdo0X}
}
