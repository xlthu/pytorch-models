import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seq2seq
from torchseq import transforms
from torchseq.datasets.tatoeba import Tatoeba
from torchseq.utils.collate_fn import pad_sequence
from utils.average_meter import AverageMeter

torch.backends.cudnn.benchmark = True

START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

parser = argparse.ArgumentParser(
    description='DL Benchmark based on Pytorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Number of data loading workers')
parser.add_argument('--epochs', default=300, type=int,
                    help='Number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Manual epoch to start training')

parser.add_argument('--atten', default="general", type=str, choices=["dot", "general", "concat"],
                    help='Attention mechanism')
parser.add_argument('--embed', default=[256, 256], type=int, nargs=2,
                    help='Word embedding dimension')
parser.add_argument('--hd', '--hidden-dim', default=[256, 256], type=int, nargs=2,
                    help='Number of hidden units per layer')
parser.add_argument('--nl', '--nlayers', default=[2, 2], type=int, nargs=2,
                    help='Number of layers')
parser.add_argument('--dropout', default=[0.1, 0.1], type=float, nargs=2,
                    help='Dropout applied to layers (0 = no dropout)')

parser.add_argument('--ratio', '--teacher-forcing-ratio', default=0.5, type=float,
                    help='Teacher forcing ratio')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='Mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('-t', '--translate', default='', type=str,
                    help='Translate')
parser.add_argument('--max-len', default=50, type=int,
                    help='The maximum length for translation')
parser.add_argument('--save-dir', default='./saves', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='Data directory')

def Tatoeba_collate_fn(batch):
    data = pad_sequence([item[0] for item in batch])
    data_mask = pad_sequence([torch.ones(len(item[0])) for item in batch])

    target = pad_sequence([item[1] for item in batch])

    return (data, data_mask, target)


def getCriterion():
    """Criterion (Loss function) for training and validation"""
    def loss_func(output, target, batch_size):
        output_for_loss = output.view(-1, output.size(2)) # seq_len * N, ntokens
        target_for_loss = target.view(-1) # seq_len * N
        loss = F.cross_entropy(output_for_loss, target_for_loss, size_average=False, ignore_index=0)
        return loss / batch_size
    return loss_func


def getOptimizer(model, lr, weight_decay):
    """Optimizer for training"""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(train_loader, model: seq2seq.Seq2seq, criterion, optimizer, epoch, teacher_forcing_ratio):
    """Run one train epoch"""
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    for i, batch in enumerate(train_loader):
        # data: seq_len, N
        # data_mask: seq_len, N
        # target: seq_len, N
        data, data_mask, target = batch
        target = target.cuda(non_blocking=True)
        data_mask = data_mask.cuda(non_blocking=True)
        data = data.cuda()

        batch_size = data.size(1)
        target_len = target.size(0)
        
        # Forward
        # Encoder
        source_hs, hidden = model.encoder(data)
        # Decoder
        ctx = None
        hidden = model.transformHidden(hidden)
        
        outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        x = target[0]
        for j in range(1, target_len):
            output, hidden, ctx = model.decoder(x, hidden, ctx, source_hs, data_mask)
            outputs.append(output)

            with torch.no_grad():
                if use_teacher_forcing:
                    x = target[j]
                else:
                    topi = torch.topk(output, 1, dim=1)[1] # N, 1
                    x = topi.squeeze() # N

        outputs = torch.stack(outputs) # seq_len, N, n_tokens
        loss = criterion(outputs, target[1:], batch_size)

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

def translate(model, source, max_len):
    # TODO
    pass

def validate(val_loader, model, criterion):
    """Run evaluation"""
    losses = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # data: seq_len, N
            # data_mask: seq_len, N
            # target: seq_len, N
            data, data_mask, target = batch
            target = target.cuda(non_blocking=True)
            data_mask = data_mask.cuda(non_blocking=True)
            data = data.cuda()

            batch_size = data.size(1)
            target_len = target.size(0)

            # Forward
            # Encoder
            source_hs, hidden = model.encoder(data)
            # Decoder
            ctx = None
            hidden = model.transformHidden(hidden)

            outputs = []
            x = target[0]
            for j in range(1, target_len):
                output, hidden, ctx = model.decoder(x, hidden, ctx, source_hs, data_mask)
                outputs.append(output)

                topi = torch.topk(output, 1, dim=1)[1] # N, 1
                x = topi.squeeze() # N

            outputs = torch.stack(outputs) # seq_len, N, n_tokens
            
            # Measure loss
            loss = criterion(outputs, target[1:], batch_size)
            losses.update(loss.item(), batch_size)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), loss=losses))

    print(' * Loss {loss.avg:.3f}'.format(loss=losses))
    return losses.avg


def saveCheckpoint(state, is_better, filename='model.pth.tar'):
    """Save the training model"""
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    if is_better:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


def adjustLearningRate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = args.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def getTatoebaDataset(root, split="train"):
    return Tatoeba("./data", lang="deu", split=split,
                start_token=START_TOKEN, end_token=END_TOKEN, unk_token=UNK_TOKEN,
                transform=transforms.ToTensor(), download=(split == "train"))


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    best_val_loss = 10000

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Data Preparation
    train_dataset = getTatoebaDataset(args.data_dir, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=Tatoeba_collate_fn,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        getTatoebaDataset(args.data_dir, split="val"),
        batch_size=args.batch_size, shuffle=False, collate_fn=Tatoeba_collate_fn,
        num_workers=args.workers, pin_memory=True)

    model = seq2seq.Seq2seq(n_tokens=[len(train_dataset.vocab_lang), len(train_dataset.vocab_eng)], 
                            embed_dim=args.embed, hidden_dim=args.hd, n_layers=args.nl,
                            dropout=args.dropout, atten_method=args.atten)
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

    if args.translate:
        translate(model, args.translate, args.max_len)
    elif args.evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            lr = adjustLearningRate(optimizer, epoch)
            print("Epoch: [{0}] Learning Rate {1:.5f}".format(epoch, lr))

            # Train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args.ratio)

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
