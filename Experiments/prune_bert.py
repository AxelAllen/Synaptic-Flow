import os
import torch
import pickle
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
    default_data_collator
)
from datasets import load_dataset, load_metric
from prune import prune_loop


def run(args):

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)


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
        wnli = load_dataset('glue', 'wnli')
        #ax = load_dataset('glue', 'ax')
        glue = {'cola': cola, 'sst2': sst2, 'mrpc': mrpc, 'qqp': qqp, 'stsb': stsb, 'mnli': mnli, 'qnli': qnli, 'rte': rte,
               'wnli': wnli} #, 'ax': ax


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
                                  sparsity=sparsity)
        #prune_results.update({task_name: prune_result})



        result_dir = os.path.join(args.result_dir, f"{task_name}")
        ckpt_dir = os.path.join(args.result_dir, "checkpoints")
        model.save_pretrained(save_directory=os.path.join(ckpt_dir, f"{task_name}"))

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