"""
Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import vgg
import resnet
from utils.average_meter import AverageMeter

import numpy as np

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='DL Benchmark based on Pytorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Number of data loading workers')
parser.add_argument('--epochs', default=300, type=int,
                    help='Number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Manual epoch to start training')
parser.add_argument('-a', '--arch', default='vgg', type=str, choices=['vgg', 'resnet'],
                    help='Model architecture')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='Mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='Learning rate at epoch 0')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='Weight decay')

parser.add_argument('--print-freq', default=20, type=int,
                    help='Print frequency')
parser.add_argument('-m', '--model', default='', type=str,
                    help='Load model at PATH')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='Evaluate model on validation set')
parser.add_argument('--save-dir', default='./saves', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='Data directory')


def getCriterion():
    """Criterion (Loss function) for training and validation"""
    return nn.CrossEntropyLoss()


def getOptimizer(model, lr, momentum, weight_decay):
    """Optimizer for training"""
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def train(train_loader, model, criterion, optimizer, epoch):
    """Run one train epoch"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda()

        # Forward
        output = model(input)
        loss = criterion(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # Print Training Information
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """Run evaluation"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda()

            # Forward
            output = model(input)

            # Measure accuracy and record loss
            loss = criterion(output, target)
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def saveCheckpoint(state, is_better, filename='model.pth.tar'):
    """Save the training model"""
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    if is_better:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


def adjustLearningRate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def getCifar10Dataset(root, isTrain=True):
    """Cifar-10 Dataset"""
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    if isTrain:
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    return datasets.CIFAR10(root=root, train=isTrain, transform=trans, download=isTrain)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    best_prec1 = 0

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Data Preparation
    train_loader = torch.utils.data.DataLoader(
        getCifar10Dataset(args.data_dir, isTrain=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        getCifar10Dataset(args.data_dir, isTrain=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.arch == "vgg":
        model = vgg.VGG16(input_size=(3, 32, 32), num_classes=10)
    elif args.arch == "resnet":
        model = resnet.Resnet18(input_size=(3, 32, 32), num_classes=10)
    else:
        raise RuntimeError("Unsupported model architecture")
    model.cuda()

    # Load model or initialize weights
    if args.model:
        if os.path.isfile(args.model):
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Load model '{}' @ epoch {}".format(
                args.model, args.start_epoch))
        else:
            raise ValueError("=> No model found at '{}'".format(args.model))
    else:
        model.initializeWeights()

    # Criterion and optimizer
    criterion = getCriterion().cuda()
    optimizer = getOptimizer(model, args.lr, args.momentum, args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            lr = adjustLearningRate(optimizer, epoch)
            print("Epoch: [{0}] Learning Rate {1:.3f}".format(epoch, lr))

            # Train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # Evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # Remember best prec@1
            is_better = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # Save checkpoint every epoch
            saveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_better, filename=os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
