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
from transformers import AdamW, get_scheduler, EvalPrediction, Trainer, TrainingArguments, default_data_collator

def train_glue(model, dataloader, optimizer, lr_scheduler, device, epoch, verbose, log_interval=10):
    model.train()

    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device)
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
            batch = batch.to(device)
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

def pre_train_eval_loop_glue(models, datasets, tokenizer, device, args, use_wandb=False):
    train_results = {}
    for (task, model), (_, dataset_dict) in zip(models.items(), datasets.items()):
        if task == "mnli":
            train_data, eval_data = dataset_dict["train"], dataset_dict["validation_matched"]
        else:
            train_data, eval_data = dataset_dict["train"], dataset_dict["validation"]

        is_regression = task == "stsb"
        metric = load_metric("glue", task)

        ## Set-up training ##
        num_update_steps_per_epoch = len(train_data) / args.train_batch_size
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

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


        ## Transformer Trainer Arguments ##
        training_args = TrainingArguments(
            output_dir="Results/bert/glue",
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.test_batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_train_epochs=args.pre_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=None,
            greater_is_better=None,
            report_to=["wandb"],
            resume_from_checkpoint=None,
            run_name=None
        )
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=default_data_collator,
                          train_dataset=train_data,
                          eval_dataset=eval_data,
                          tokenizer=tokenizer,
                          compute_metrics=compute_metrics,
                          optimizers=(optimizer, lr_scheduler))


        ## train ##
        print(f"Fine-tuning on {task} dataset.")
        train_result = trainer.train()
        train_results.update({task: train_result})


        '''
        for epoch in tqdm(range(args.pre_epochs)):
            GPUtil.showUtilization()
            train_loss = train_glue(model, train_loader, optimizer, lr_scheduler, device, epoch, args.verbose)
            if use_wandb:
                wandb.log({"pre_train_loss": train_loss})
            
            eval_metrics = eval_glue(task, model, eval_loader, device, args.verbose)
            if use_wandb:
                wandb.log({"pre_eval_metrics": eval_metrics})
        '''
            #row = [train_loss, test_loss, accuracy1, accuracy5]
            #lr_scheduler.step()
            #rows.append(row)
        #columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    # return pd.DataFrame(rows, columns=columns)
    return train_results


def post_train_eval_loop_glue(models, dataloaders, tokenizer, device, args, use_wandb=False):
    train_results = {}
    for (task, model), (_, dataloader_dict) in zip(models.items(), dataloaders.items()):
        if task == "mnli":
            train_loader, eval_loader = dataloader_dict["train"], dataloader_dict["validation_matched"]
        else:
            train_loader, eval_loader = dataloader_dict["train"], dataloader_dict["validation"]

        is_regression = task == "stsb"
        metric = load_metric("glue", task)

        ## Set-up training ##
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

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


        ## Transformer Trainer Arguments ##
        training_args = TrainingArguments(
            output_dir="Results/bert/glue",
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.test_batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_train_epochs=args.pre_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=None,
            greater_is_better=None,
            report_to=["wandb"],
            resume_from_checkpoint=None,
            run_name=None
        )
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=default_data_collator,
                          train_dataset=train_loader,
                          eval_dataset=eval_loader,
                          tokenizer=tokenizer,
                          compute_metrics=compute_metrics,
                          optimizers=(optimizer, lr_scheduler))
        ## train ##
        train_result = trainer.train()
        train_results.update({task: train_result})

    return train_results


