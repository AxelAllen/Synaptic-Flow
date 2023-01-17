import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utils import load
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    ReformerForSequenceClassification,
    ReformerTokenizer,
    ReformerConfig,
    PretrainedConfig,
    default_data_collator,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_metric
import wandb
from prune import prune_loop
#from train_glue import pre_train_eval_loop_glue, post_train_eval_loop_glue


def run(args):

    if args.wandb:
        wandb.login()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    print(f"available devices: {torch.cuda.device_count()}")
    print(f"device: {device}")


    ## Data ##
    print('Loading GLUE dataset.')

    ## load GLUE datasets ##
    if len(args.glue_dataset_list) > 0:
        glue = {}
        for dataset_name in args.glue_dataset_list:
            dataset = load_dataset('glue', dataset_name)
            glue.update({dataset_name: dataset})
    else:
        cola = load_dataset('glue', 'cola')
        sst2 = load_dataset('glue', 'sst2')
        mrpc = load_dataset('glue', 'mrpc')
        qqp = load_dataset('glue', 'qqp')
        stsb = load_dataset('glue', 'stsb')
        mnli = load_dataset('glue', 'mnli')
        qnli = load_dataset('glue', 'qnli')
        rte = load_dataset('glue', 'rte')
        glue = {'cola': cola, 'sst2': sst2, 'mrpc': mrpc, 'qqp': qqp, 'stsb': stsb, 'mnli': mnli, 'qnli': qnli, 'rte': rte}

    ## map task to keys ##
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    ## max length ##
    '''
    max_lengths = {}
    columns = ["sentence", "premise", "hypothesis", "sentence1", "sentence2", "question", "question1", "question2"]
    for name, dataset_dict in glue.items():
        max_length = 0
        for _, dataset_split in dataset_dict.items():
            for col in columns:
                if col in dataset_split.features:
                    for sentence in dataset_split[col]:
                        sent_len = len(sentence)
                        if sent_len > max_length:
                            max_length = sent_len
        max_lengths.update({name: max_length})
    '''
    ## labels ##
    labels = {}
    for name, dataset in glue.items():
        if name == 'stsb':
            label_list = None
            num_labels = 1
        else:
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)
        labels.update({name: {"label_list": label_list, "num_labels": num_labels}})

    train_results = {}
    eval_results = {}
    prune_results = {}
    ## Models ##
    models = {}
    for (task_name, label_dict), (_, dataset) in zip(labels.items(), glue.items()):
        if args.wandb:
            wandb.init(
                project=f"{args.projectid}",
                name=f"{task_name}_{args.expid}",
                group=f"{args.groupid}",
                config=vars(args)
            )
        is_regression = task_name == "stsb"

        if args.model == "Bert":
            config = AutoConfig.from_pretrained(
                "bert-base-uncased",
                num_labels=label_dict["num_labels"],
                finetuning_task=task_name
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                config=config
            )
        elif args.model == "Reformer":
            config = AutoConfig.from_pretrained(
                "google/reformer-crime-and-punishment",
                num_labels=label_dict["num_labels"],
                finetuning_task=task_name,
                axial_pos_embds=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "google/reformer-crime-and-punishment"
            )
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(
                "google/reformer-crime-and-punishment",
                config=config
            )
        else:
            print("Invalid model name: initializing Bert-Base-Uncased instead.")
            config = AutoConfig.from_pretrained(
                "bert-base-uncased",
                num_labels=label_dict["num_labels"],
                finetuning_task=task_name
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                config=config
            )


        sentence1_key, sentence2_key = task_to_keys[task_name]
        padding = "max_length"

        label_to_id = None
        if (hasattr(model.config, "label2id") and
            model.config.label2id != PretrainedConfig(num_labels=label_dict["num_labels"]).label2id
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_dict["label_list"])):
                label_to_id = {i: int(label_name_to_id[label_dict["label_list"][i]]) for i in range(label_dict["num_labels"])}
            else:
                print(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_dict['label_list']))}."
                    "\nIgnoring the model labels as a result.",
                )

        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif not is_regression:
            model.config.label2id = {l: i for i, l in enumerate(label_dict["label_list"])}
            model.config.id2label = {id: label for label, id in config.label2id.items()}

        '''
        if max_lengths[task_name] > tokenizer.model_max_length:
            print(
                f"The max_seq_length passed ({max_lengths[task_name]}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(max_lengths[task_name], tokenizer.model_max_length)
        '''
        max_seq_length = args.max_sequence_length

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        datasets = dataset.map(
            preprocess_function,
            batched=True
        )
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
        #test_dataset = datasets["test_matched" if task_name == "mnli" else "test"]
        metric = load_metric("glue", task_name)

        ## Prune the model ##
        dataloader = DataLoader(dataset=train_dataset,
                                shuffle=True,
                                collate_fn=default_data_collator,
                                batch_size=args.train_batch_size,
                                pin_memory=True)
        loss_func = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        sparsity = 10 ** (-float(args.compression))
        prune_results, unit_scores, avg_layer_scores = prune_loop(model=model,
                                  prune_class=args.pruner,
                                  dataloader=dataloader,
                                  loss=loss_func,
                                  device=device,
                                  epochs=args.prune_epochs,
                                  schedule=args.compression_schedule,
                                  scope=args.mask_scope,
                                  sparsity=sparsity, #args.compression
                                  use_wandb=args.wandb)
        #prune_results.update({task_name: prune_result})

        result_dir = os.path.join(args.result_dir, f"{task_name}")

        try:
            os.makedirs(result_dir)
        except FileExistsError:
            print("Overwriting results for task '{}' within Experiment '{}' with expid '{}'".format(task_name, args.experiment, args.expid))

        with open(f"{result_dir}/prune-results.pickle", "wb") as w:
            pickle.dump(prune_results, w, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{result_dir}/unit-scores.pickle", "wb") as w:
            pickle.dump(unit_scores, w, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{result_dir}/layerwise-scores.pickle", "wb") as w:
            pickle.dump(avg_layer_scores, w, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        with open(f"{result_dir}/importance-scores.pickle", "wb") as w:
            pickle.dump(importance_scores, w, protocol=pickle.HIGHEST_PROTOCOL)

        
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
        '''
        if args.pre_epochs > 0:
            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result


            report_to = ["wandb"] if args.wandb else "none"
            training_args = TrainingArguments(
                output_dir=f"Results/bert/glue/{task_name}",
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                do_predict=False,
                evaluation_strategy="epoch",
                save_strategy="no",
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.test_batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_train_epochs=args.pre_epochs,
                lr_scheduler_type=args.lr_scheduler_type,
                seed=args.seed,
                report_to=report_to
                #load_best_model_at_end=True,
                #metric_for_best_model=None,
                #greater_is_better=None,
                #resume_from_checkpoint=None,
                #run_name=None
            )
            trainer = Trainer(model=model,
                              args=training_args,
                              data_collator=default_data_collator,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              tokenizer=tokenizer,
                              compute_metrics=compute_metrics)
            print("*** Train ***")
            train_result = trainer.train()
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            train_results.update({task_name: train_result})

            print("*** Evaluate ***")
            tasks = [task_name]
            eval_datasets = [eval_dataset]
            combined = {}
            if task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(datasets["validation_mismatched"])
            for eval_data, task in zip(eval_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=eval_data)
                metrics["eval_samples"] = len(eval_data)

                if "mnli" in task:
                    if task == "mnli-mm":
                        metrics = {k + "_mm": v for k, v in metrics.items()}
                    combined.update(metrics)
                    eval_results.update({task_name: combined})
                else:
                    eval_results.update({task_name: metrics})
            if args.wandb:
                wandb.finish()

        with open(f"{args.result_dir}/train-results.pickle", "wb") as w:
            pickle.dump(train_results, w, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{args.result_dir}/eval-results.pickle", "wb") as w:
            pickle.dump(eval_results, w, protocol=pickle.HIGHEST_PROTOCOL)

