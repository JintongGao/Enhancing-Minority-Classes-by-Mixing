import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--root', default='./data/', help='dataset setting')
parser.add_argument('--dataset', default='cifar10', help='dataset setting', choices=('cifar100', 'cifar10', 'Imagenet-LT', 'iNat18'))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes ')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')

parser.add_argument('--loss_type', default='ERM', type=str, help='loss type / method', choices=('ERM', 'LDAM', 'BALMS'))
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader',
                    choices=('None', 'DRW'))
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--w', default=0.05, type=float, help='feature weight')
parser.add_argument('--u', default=1.0, type=float, help='binomial_s')

parser.add_argument('--data_aug', default="OT", type=str, help='data augmentation type',
                    choices=('vanilla', 'OT'))
parser.add_argument('--mixup_prob', default=0.5, type=float, help='mixup probability')
parser.add_argument('--start_data_aug', default=3, type=int, help='start epoch for aug')
parser.add_argument('--end_data_aug', default=3, type=int, help='how many epochs to turn off aug')
parser.add_argument('--weighted_alpha', default=1, type=float, help='weighted alpha for sampling probability (q(1,k))')
parser.add_argument('--alpha', default=4, type=float, help='hyperparam for beta distribution')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
