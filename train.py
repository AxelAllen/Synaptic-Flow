import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sam.sam import SAM

def train(model, loss, optimizer, dataloader, gpu_id, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if torch.cuda.is_available() and gpu_id is not None:
            data, target = data.cuda(gpu_id, non_blocking=True), target.cuda(gpu_id, non_blocking=True)
        elif torch.cuda.is_available():
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)

        if isinstance(optimizer, SAM):
            train_loss.mean().backward()
            optimizer.first_step(zero_grad=True)
            loss(model(data), target).mean().backward()
            optimizer.second_step(zero_grad=True)
        else:
            train_loss.backward()
            optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, gpu_id, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available() and gpu_id is not None:
                data, target = data.cuda(gpu_id, non_blocking=True), target.cuda(gpu_id, non_blocking=True)
            elif torch.cuda.is_available():
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, train_sampler, gpu_id, epochs, verbose, distributed):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, gpu_id, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        if distributed:
            train_sampler.set_epoch(epoch)
        train_loss = train(model, loss, optimizer, train_loader, gpu_id, epoch, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, gpu_id, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


