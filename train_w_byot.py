import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT

from models import resnet_cifar_byot

from utils import config, update_config, create_logger
from utils import ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion

def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)
    parser.add_argument('--save-path', default='./runs', type=str)
    parser.add_argument('--data-dir', default='./data/cifar100', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='dataset choice')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_ratio', default=0.1, type=float, help='imbalance factor')

    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=2e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')

    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--model_dir', default='./ckps')

    parser.add_argument('--model', metavar='ARCH', default='resnet32_byot',
                        help='model architecture')
    parser.add_argument('--loss_name_list', default=['CE'], nargs='+',
                        help='which loss type to use on shallow fc layer')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    args = parser.parse_args()
    # update_config(config, args)

    return args

best_acc1 = 0
its_ece = 100


def main():
    args = parse_args()
    if len(args.loss_name_list)==3:
        args.use_byot = True
    else:
        args.use_byot = False
    if args.dataset.startswith('imagenet'):
        dataset = args.dataset
    elif args.dataset.startswith(''):
        dataset = '_'.join([args.dataset, args.imb_type, (str)(args.imb_ratio)])

    loss_type = '_'.join(args.loss_name_list)
    use_byot = (str)(args.use_byot)
    aplha_beta = '_'.join([(str)(args.temperature), (str)(args.alpha), (str)(args.beta)])
    save_path = args.save_path = os.path.join(args.save_path, dataset, args.model, use_byot, loss_type, aplha_beta)
    print(save_path)
    # return
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    args.logger_file = os.path.join(save_path, 'log_train.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args, ngpus_per_node)

def main_worker(args, ngpus_per_node):
    global best_acc1, its_ece
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if args.dataset == 'cifar10':
        dataset = CIFAR10_LT(root=args.data_dir, imb_factor=args.imb_ratio,
                             batch_size=args.batch_size, num_works=args.workers)
        args.num_classes = 10
        args.head_class_idx = [0, 3]
        args.med_class_idx = [3, 7]
        args.tail_class_idx = [7, 10]

    elif args.dataset == 'cifar100':
        dataset = CIFAR100_LT(root=args.data_dir, imb_factor=args.imb_ratio,
                              batch_size=args.batch_size, num_works=args.workers)
        args.num_classes = 100
        args.head_class_idx = [0, 36]
        args.med_class_idx = [36, 71]
        args.tail_class_idx = [71, 100]

    cls_num_list = dataset.cls_num_list

    model = getattr(resnet_cifar_byot, args.model)(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cuda(device)

    train_loader = dataset.train_instance
    val_loader = dataset.eval

    criterion_1, criterion_2, criterion_3 = get_loss(args, cls_num_list, device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion_1, criterion_2, criterion_3, optimizer, epoch, args)
        acc1, ece = validate(val_loader, model, criterion_3, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logging.info('Best Prec@1: %.3f%% ECE: %.3f%%\n' % (best_acc1, its_ece))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'best_acc1': best_acc1,
            'its_ece': its_ece,
        }, is_best, args.save_path)


def train(train_loader, model, criterion_1, criterion_2, criterion_3, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    losses1_kd = AverageMeter()
    losses2_kd = AverageMeter()
    feature_losses_1 = AverageMeter()
    feature_losses_2 = AverageMeter()
    total_losses = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()


    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i > end_steps:
            break
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output, middle_output1, middle_output2, final_feat, middle_feat1, middle_feat2 = model(input)

        loss = criterion_3(output, target)
        losses.update(loss.item(), input.size(0))
        if args.use_byot:
            middle1_loss = criterion_1(middle_output1, target)
            middle1_losses.update(middle1_loss.item(), input.size(0))
            middle2_loss = criterion_2(middle_output2, target)
            middle2_losses.update(middle2_loss.item(), input.size(0))

            temp4 = output / args.temperature
            temp4 = torch.softmax(temp4, dim=1)

            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature ** 2)
            losses1_kd.update(loss1by4, input.size(0))
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature ** 2)
            losses2_kd.update(loss2by4, input.size(0))

            feature_loss_1 = feature_loss_function(middle_feat1, final_feat.detach())
            feature_losses_1.update(feature_loss_1, input.size(0))
            feature_loss_2 = feature_loss_function(middle_feat2, final_feat.detach())
            feature_losses_2.update(feature_loss_2, input.size(0))

            total_loss = (1 - args.alpha) * (loss + middle1_loss + middle2_loss) + \
                         args.alpha * (loss1by4 + loss2by4) + \
                         args.beta * (feature_loss_1 + feature_loss_2)
            total_losses.update(total_loss.item(), input.size(0))

            middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    class_num = torch.zeros(args.num_classes).cuda()
    correct = torch.zeros(args.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output, middle_output1, middle_output2, final_feat, middle_feat1, middle_feat2 = model(input)

            loss = criterion(output, target)

            middle1_loss = criterion(middle_output1, target)
            middle1_losses.update(middle1_loss.item(), input.size(0))
            middle2_loss = criterion(middle_output2, target)
            middle2_losses.update(middle2_loss.item(), input.size(0))

            middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, args.num_classes)
            predict_one_hot = F.one_hot(predicted, args.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logging.info("Loss {loss.avg:.3f}\t"
                     "Prec@1 {top1.avg:.3f}\t".format(
            loss=losses,
            top1=top1,
            ))

        acc_classes = correct / class_num
        head_acc = acc_classes[args.head_class_idx[0]:args.head_class_idx[1]].mean() * 100
        med_acc = acc_classes[args.med_class_idx[0]:args.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[args.tail_class_idx[0]:args.tail_class_idx[1]].mean() * 100

    cal = calibration(true_class, pred_class, confidence, num_bins=15)
    return top1.avg, cal['expected_calibration_error'] * 100


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    # if args.cos:
    #     lr_min = 0
    #     lr_max = args.lr
    #     lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / args.num_epochs * 3.1415926535))
    # else:
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.01
    elif epoch > 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def kd_loss_function(output, target_output, args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss(args, cls_num_list, device):
    from loss.SoftmaxLoss import CrossEntropyLoss as ce
    from loss.FocalLoss import FocalLoss as focal
    from loss.LDAMLoss import LDAMLoss as ldam
    from loss.BalancedSoftmaxLoss import BalancedSoftmax as balanced_loss
    if args.loss_name_list == ['CE']:
        return None, None, nn.CrossEntropyLoss().cuda()
    elif args.loss_name_list == ['balanced']:
        return None, None, balanced_loss(cls_num_list).to(device)
    else:
        loss_name_list = args.loss_name_list

        arg_list = {
            'CE': {'cls_num_list': None, 'reweight_CE': False},
            'focal': {'gamma': 1.0},
            'LDAM': {'cls_num_list': cls_num_list},
            'balanced': {'cls_num_list': cls_num_list},
            'effective': {'cls_num_list': cls_num_list, 'reweight_CE': True}
        }

        loss_dict = {
            'CE': ce,
            'focal': focal,
            'LDAM': ldam,
            'balanced': balanced_loss,
            'effective': ce
        }
        return [loss_dict[loss_name](**arg_list[loss_name]).to(device) for loss_name in loss_name_list]


if __name__ == '__main__':
    main()
