import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sam.sam import SAM
import GPUtil
import wandb

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(train_loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        optimizer.zero_grad()
        if isinstance(optimizer, SAM):
            train_loss.mean().backward()
            optimizer.first_step(zero_grad=True)
            loss(model(data), target).mean().backward()
            optimizer.second_step()
        else:
            train_loss.backward()
            optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc1: {:.6f}\tAcc5: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item(), top1.avg, top5.avg))
    return losses.avg, top1.avg, top5.avg

def eval(model, loss_func, dataloader, device, verbose):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

    average_loss = losses.avg
    accuracy1 = top1.avg
    accuracy5 = top5.avg
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: ({:.2f}%), Top 5 Accuracy: ({:.2f}%)'.format(
            average_loss, accuracy1, accuracy5))
    return average_loss, accuracy1, accuracy5

def pre_train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, use_wandb=False):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        GPUtil.showUtilization()
        train_loss, train_acc1, train_acc5 = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        if use_wandb:
            wandb.log({"pre_train_loss": train_loss, "pre_train_acc1": train_acc1, "pre_train_acc5": train_acc5})
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        if use_wandb:
            wandb.log({"pre_eval_loss": test_loss, "pre_eval_acc1": accuracy1, "pre_eval_acc5": accuracy5})
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)

def post_train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, use_wandb=False):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        GPUtil.showUtilization()
        train_loss, train_acc1, train_acc5  = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        if use_wandb:
            wandb.log({"post_train_loss": train_loss, "post_train_acc1": train_acc1, "post_train_acc5": train_acc5})
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        if use_wandb:
            wandb.log({"post_eval_loss": test_loss, "post_eval_acc1": accuracy1, "post_eval_acc5": accuracy5})
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
