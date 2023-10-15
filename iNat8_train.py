# reference code: https://github.com/naver-ai/cmo

import argparse
import os
import random
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from imbalance_data.lt_data import LT_Dataset
from losses import LDAMLoss, BalancedSoftmaxLoss
from opts import parser
from tensorboardX import SummaryWriter
import warnings
import math
from torch.nn import Parameter
import torch.nn.functional as F
from util import *
from models.resnet import *
import ot

d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
best_acc1 = 0

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_type, args.train_rule, args.data_aug, str(args.epochs), str(args.w), str(args.weighted_alpha), str(args.batch_size), str(args.u)])
    prepare_folders(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda
    global confusion_cf

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 8142
    model = resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    if args.loss_type == 'LDAM':
        model.fc = NormedLinear(num_ftrs, num_classes)
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()  
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
  
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True
    
    # Data loading code
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    ])
    transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
    ])

    train_dataset = LT_Dataset(args.root, "i2018_LT/iNaturalist18_train.txt", transform_train)
    val_dataset = LT_Dataset(args.root, "i2018_LT/iNaturalist18_val.txt", transform_val)

    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 8142

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    #print('cls num list:')
    #print(cls_num_list)
    #print(sum(cls_num_list))
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    weighted_train_loader = None
    weighted_cls_num_list = [0] * num_classes

    if args.data_aug == 'OT':
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler)
    print("data loaders loaded")

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    #start_time = time.time()
    print("Training started!")

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'ERM':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BALMS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return
        
        # select method and train
        if (np.random.binomial(n=1, p=(epoch/args.epochs)**args.u))<1:  
            train_random(val_loader, model, criterion, optimizer, epoch, args, log_training,
              tf_writer, weighted_train_loader)
        else:
            # get confusion matrix    
            cf = confusion_cf
            cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
            for l1 in range(num_classes):
                cf[l1][l1] = 0
            
            train_OTmix(val_loader, model, criterion, optimizer, epoch, cf, args, log_training, tf_writer, weighted_train_loader)
        
        # evaluate on validation set
        acc1, confusion_cf = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, epoch + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write("It took {} to execute the program".format(hms_string(end_time - start_time)) + '\n')
    log_testing.flush()

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def train_random(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer=None,
          weighted_train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        inverse_iter = iter(weighted_train_loader)

    for i, (input, target) in enumerate(train_loader):
        if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                input2, target2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                input2, target2 = next(inverse_iter)

            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # Data augmentation
        r = np.random.rand(1)

        if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) \
                and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.alpha, args.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            _, output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)

        else:
            _, output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def train_OTmix(train_loader, model, criterion, optimizer, epoch, cf,  args, log, tf_writer=None,
          weighted_train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        weighted_train_loader = iter(weighted_train_loader)

    for i, (input, target) in enumerate(train_loader):
        if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                input2, target2 = next(weighted_train_loader)
            except:
                weighted_train_loader = iter(weighted_train_loader)
                input2, target2 = next(weighted_train_loader)

            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # Data augmentation
        r = np.random.rand(1)

        if args.data_aug == 'OT' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
                  
            B, c, w, h = input.size()
            f, _ = model(input)
            f2, _ = model(input2)
            
            tar1 = target.cpu().numpy()
            tar2 = target2.cpu().numpy()
            
            # get confusion
            cf_scale = [[0 for t2 in range(B)] for t1 in range(B)]
            for q1 in range(B):
                for q2 in range(B):
                    cf_scale[q1][q2] = cf[tar2[q1]][tar1[q2]]
            cf_scale = torch.from_numpy(np.array(cf_scale)).cuda(args.gpu)
            
            # get distance
            x_col = f2.unsqueeze(-2)
            y_lin = f.unsqueeze(-3)
            C_feature = 1-d_cosine(x_col, y_lin)
            
            # get cost Cij
            M = args.w1 * C_feature + (1 - args.w1) * (1 - cf_scale)  # (0.05, 0.95)
            
            # get Tij
            x_points = f2.shape[-2]
            y_points = f.shape[-2]
            if f.dim() == 2:
                batch_size = 1
            else:
                batch_size = x.shape[0]
            a = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1).cuda().squeeze()                 
            b = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1).cuda().squeeze()
            T = ot.sinkhorn(a, b, M, reg = 0.1)
            
            input3 = torch.zeros_like(input2)
            target3 = torch.zeros_like(target)
            T = T.cpu().detach().numpy()
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            for k in range(B):
                index = np.argmax(T[k], axis=None)
                input3[k, :, :, :] = input[index, :, :, :]
                input3[k, :, bbx1:bbx2, bby1:bby2] = input2[k, :, bbx1:bbx2, bby1:bby2]
                target3[k] = target[index]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            _, output = model(input3)
            loss = criterion(output, target3) * lam + criterion(output, target2) * (1. - lam)

        else:
            _, output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            _, output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)

        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20
        print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
              float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
              float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)

    return top1.avg, cf

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 130:
        lr = args.lr * 0.001
    elif epoch > 80:
        lr = args.lr * 0.01
    elif epoch > 30:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
