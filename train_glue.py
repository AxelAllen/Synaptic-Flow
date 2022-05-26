import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sam.sam import SAM
from datasets import load_metric
import GPUtil
import wandb
from Utils import generator
from transformers import AdamW, get_scheduler

def train_glue(model, dataloader, optimizer, lr_scheduler, device, epoch, verbose, log_interval=10):
    model.train()

    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    avg_train_loss = total_loss / len(dataloader)

    if verbose & (step % log_interval == 0):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, step * len(batch), len(dataloader.dataset),
            100. * step / len(dataloader), loss.item()))
    return avg_train_loss

def eval_glue(task, model, dataloader, device, verbose):
    model.eval()
    metric = load_metric("glue", task)
    is_regression = task == "stsb"
    eval_metrics = []
    if task == "mnli":
        pass
    else:
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            references = None
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            eval_metric = metric.compute()
            eval_metrics.append(eval_metric)
    if verbose:
        pass
        '''print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))'''
    return eval_metrics

def pre_train_eval_loop_glue(models, dataloaders, device, args, use_wandb=False):
    for (task, model), (_, dataloader_dict) in zip(models.items(), dataloaders.items()):
        if task == "stsb":
            continue
        if task == "mnli":
            train_loader, eval_loader = dataloader_dict["train"], dataloader_dict["validation_matched"]
        else:
            train_loader, eval_loader = dataloader_dict["train"], dataloader_dict["validation"]

        ## Train ##
        print(f"Fine-tuning on {task} dataset.")

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = args.pre_epochs * num_update_steps_per_epoch

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        for epoch in tqdm(range(args.pre_epochs)):
            # GPUtil.showUtilization()
            train_loss = train_glue(model, train_loader, optimizer, lr_scheduler, device, epoch, args.verbose)
            if use_wandb:
                wandb.log({"pre_train_loss": train_loss})
            '''
            eval_metrics = eval_glue(task, model, eval_loader, device, args.verbose)
            if use_wandb:
                wandb.log({"pre_eval_metrics": eval_metrics})
            '''
            #row = [train_loss, test_loss, accuracy1, accuracy5]
            #lr_scheduler.step()
            #rows.append(row)
        #columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    # return pd.DataFrame(rows, columns=columns)


def post_train_eval_loop_glue(models, dataloaders, device, args, use_wandb=False):
    for (task, model), (_, dataloader_dict) in zip(models, dataloaders):
        train_loader, eval_loader, test_loader = dataloader_dict["train"], dataloader_dict["validation"], \
                                                 dataloader_dict["test"]

        ## Train ##
        print(f"Fine-tuning on {task} dataset.")
        print(f"Fine-tuning on {task} dataset.")

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = args.post_epochs * num_update_steps_per_epoch

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        for epoch in tqdm(range(args.post_epochs)):
            GPUtil.showUtilization()
            train_loss = train_glue(model, train_loader, optimizer, lr_scheduler, device, epoch, args.verbose)
            if use_wandb:
                wandb.log({"post_train_loss": train_loss})
            '''
            eval_metric = eval_glue(task, model, eval_loader, device, args.verbose)
            if use_wandb:
                wandb.log({"post_eval_metric": eval_metric})
            '''
            # row = [train_loss, test_loss, accuracy1, accuracy5]
            #lr_scheduler.step()
            # rows.append(row)
        # columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
        return 0  # pd.DataFrame(rows, columns=columns)


