import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

import aoa
from torchseq import transforms
from torchseq.datasets.cbt import CBT
from torchseq.utils.collate_fn import pad_sequence
from utils.average_meter import AverageMeter

torch.backends.cudnn.benchmark = True

UNK_TOKEN = "<unk>"

parser = argparse.ArgumentParser(
    description='DL Benchmark based on Pytorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Number of data loading workers')
parser.add_argument('--epochs', default=300, type=int,
                    help='Number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Manual epoch to start training')
parser.add_argument('--type', default='CN', type=str,
                    help='Word type (CN, NE, P, V)')

parser.add_argument('--embed', default=384, type=int,
                    help='Word embedding dimension')
parser.add_argument('--hd', '--hidden-dim', default=256, type=int,
                    help='Number of hidden units per layer')
parser.add_argument('--nl', '--nlayers', default=2, type=int,
                    help='Number of layers')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout applied to layers (0 = no dropout)')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='Mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='Learning rate at epoch 0')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='Weight decay')
parser.add_argument('--clip', default=5, type=float,
                    help='Gradient clipping')

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


def CBT_collate_fn(batch):
    batch_doc = pad_sequence([item["documents"] for item in batch])
    batch_doc_mask = pad_sequence(
        [torch.ones(len(item["documents"])) for item in batch])
    batch_doc_len = torch.tensor([len(item["documents"])
                                  for item in batch], dtype=torch.long)

    batch_query = pad_sequence([item["query"] for item in batch])
    batch_query_mask = pad_sequence(
        [torch.ones(len(item["query"])) for item in batch])

    return {
        "documents": batch_doc,
        "documents_mask": batch_doc_mask,
        "documents_len": batch_doc_len,
        "query": batch_query,
        "query_mask": batch_query_mask,
        "answer": torch.stack([item["answer"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch])
    }


def getCriterion():
    """Criterion (Loss function) for training and validation"""
    def loss_func(probs, answer, candidates):
        # probs: N, m
        # answer: N
        # candidates: N, m
        answer = answer.unsqueeze(1).expand_as(candidates)  # N, m
        mask = (answer == candidates)  # N, m

        probs = torch.masked_select(probs, mask)  # N
        assert(probs.size() == (probs.size(0), ))

        return -torch.mean(torch.log(probs))

    return loss_func


def getOptimizer(model, lr, weight_decay):
    """Optimizer for training"""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(train_loader, model, criterion, optimizer, epoch):
    """Run one train epoch"""
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    for i, batch in enumerate(train_loader):
        documents, documents_mask, documents_len = batch["documents"], batch["documents_mask"], batch["documents_len"]
        query, query_mask = batch["query"], batch["query_mask"]
        answer, candidates = batch["answer"], batch["candidates"]

        answer, candidates = answer.cuda(non_blocking=True), candidates.cuda(non_blocking=True)
        documents, documents_mask, documents_len = documents.cuda(), documents_mask.cuda(), documents_len.cuda()
        query, query_mask = query.cuda(), query_mask.cuda()

        batch_size = documents.size(1)

        # Forward
        probs = model(documents, documents_mask, documents_len,
                      query, query_mask, candidates)

        loss = criterion(probs, answer, candidates)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # Measure loss
        losses.update(loss.item(), batch_size)

        # Print Training Information
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), loss=losses))


def validate(val_loader, model, criterion):
    """Run evaluation"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            documents, documents_mask, documents_len = batch["documents"], batch["documents_mask"], batch["documents_len"]
            query, query_mask = batch["query"], batch["query_mask"]
            answer, candidates = batch["answer"], batch["candidates"]

            answer, candidates = answer.cuda(non_blocking=True), candidates.cuda(non_blocking=True)
            documents, documents_mask, documents_len = documents.cuda(), documents_mask.cuda(), documents_len.cuda()
            query, query_mask = query.cuda(), query_mask.cuda()

            batch_size = documents.size(1)

            # Forward
            probs = model(documents, documents_mask,
                          documents_len, query, query_mask, candidates)

            # Measure loss
            loss = criterion(probs, answer, candidates)
            losses.update(loss.item(), batch_size)
            prec1 = accuracy(probs, answer, candidates)
            top1.update(prec1, batch_size)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), loss=losses, top1=top1))

    print(
        ' * Loss {loss.avg:.3f}\tPrec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
    return losses.avg


def accuracy(probs, answer, candidates):
    # probs: N, m
    # answer: N
    # candidates: N, m
    """Computes the precision@1 for the specified values of k"""
    with torch.no_grad():
        batch_size = answer.size(0)
        pred_loc = torch.max(probs, dim=1)[1]  # N

        answer = answer.unsqueeze(1).expand_as(candidates)  # N, m
        mask = (answer == candidates)  # N, m
        truth = torch.max(mask, dim=1)[1]  # N

        return (pred_loc == truth).sum().item() * 100.0 / batch_size


def saveCheckpoint(state, is_better, filename='model.pth.tar'):
    """Save the training model"""
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    if is_better:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


def adjustLearningRate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def getCBTDataset(root, word_type="CN", split="train"):
    return CBT(root, word_type, split, unk_token=UNK_TOKEN, transform=transforms.ToTensor(), download=True)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    best_val_loss = 10000

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Data Preparation
    train_dataset = getCBTDataset(
        args.data_dir, word_type=args.type, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=CBT_collate_fn,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        getCBTDataset(args.data_dir, word_type=args.type, split="val"),
        batch_size=args.batch_size, shuffle=False, collate_fn=CBT_collate_fn,
        num_workers=args.workers, pin_memory=True)

    model = aoa.AoAReader(len(train_dataset.vocab),
                          args.embed, args.hd, args.nl, args.dropout)
    model.cuda()
    model.flatten_parameters()

    # Load model or initialize weights
    if args.model:
        if os.path.isfile(args.model):
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Load model '{}' @ epoch {}".format(
                args.model, args.start_epoch))
        else:
            raise ValueError("=> No model found at '{}'".format(args.model))
    else:
        model.initializeWeights()

    # Criterion and optimizer
    criterion = getCriterion()
    optimizer = getOptimizer(model, args.lr, args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            lr = adjustLearningRate(optimizer, epoch)
            print("Epoch: [{0}] Learning Rate {1:.5f}".format(epoch, lr))

            # Train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # Evaluate on validation set
            val_loss = validate(val_loader, model, criterion)

            # Remember best val loss
            is_better = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            # Save checkpoint every epoch
            saveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, is_better, filename=os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
