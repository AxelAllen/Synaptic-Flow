import torch
from torch.utils.data import DataLoader
from Utils import load
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    PretrainedConfig,
    default_data_collator,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import wandb
from train_glue import pre_train_eval_loop_glue, post_train_eval_loop_glue
from prune import prune_loop

def run(args):

    if args.wandb:
        wandb.login()
        wandb.init(
            project="synflow",
            name=f"{args.expid}",
            group=f"{args.groupid}",
            config=vars(args)
            )
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)


    ## Data ##
    print('Loading GLUE dataset.')

    ## load GLUE datasets ##
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

    # glue = {'cola': cola, 'sst2': sst2}

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

    ## labels ##
    labels = {}
    for name, dataset in glue.items():
        if name == 'stsb':
            num_labels = 1
            labels.update({name: (None, num_labels)})
        else:
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)
            labels.update({name: (label_list, num_labels)})

    ## Models ##
    models = {}
    for name, lab in labels.items():
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=lab[1], finetuning_task=name)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config).to(device)
        models.update({name: model})


    ## Preprocess datasets ##
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    glue_preprocessed = {}
    for (task_name, task), (model_name, model), (lname, label) in zip(glue.items(), models.items(), labels.items()):
        is_regression = task_name == "stsb"
        sentence1_key, sentence2_key = task_to_keys[task_name]
        max_length = max_lengths[task_name]

        ## Map labels to IDs ##
        label_to_id = None
        if (
          hasattr(model.config, "label2id")
          and model.config.label2id != PretrainedConfig(num_labels=label[1]).label2id
          and not is_regression
        ):
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label[0])):
                label_to_id = {i: label_name_to_id[label[0][i]] for i in range(label[1])}
        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in model.config.label2id.items()}
        elif not is_regression:
            model.config.label2id = {l: i for i, l in enumerate(label[0])}
            model.config.id2label = {id: label for label, id in model.config.label2id.items()}

        ## Preprocess datasets ##
        processed_dataset = task.map(
            function=preprocess_function,
            batched=True,
            # batch_size=args.batch_size,
            remove_columns=task["train"].column_names,
        )
        glue_preprocessed.update({task_name: processed_dataset})


    ## Create dataloaders ##
    dataloaders = {}

    for task_name, task in glue_preprocessed.items():
        if task_name == "mnli":
            train_dataloader = DataLoader(dataset=task["train"],
                                          shuffle=True,
                                          collate_fn=default_data_collator,
                                          batch_size=args.train_batch_size)
            eval_dataloader_matched = DataLoader(dataset=task["validation_matched"],
                                         shuffle=False,
                                         collate_fn=default_data_collator,
                                         batch_size=args.test_batch_size)
            eval_dataloader_mismatched = DataLoader(dataset=task["validation_mismatched"],
                                         shuffle=False,
                                         collate_fn=default_data_collator,
                                         batch_size=args.test_batch_size)
            test_dataloader_matched = DataLoader(dataset=task["test_matched"],
                                         shuffle=False,
                                         collate_fn=default_data_collator,
                                         batch_size=args.test_batch_size)
            test_dataloader_mismatched = DataLoader(dataset=task["test_mismatched"],
                                         shuffle=False,
                                         collate_fn=default_data_collator,
                                         batch_size=args.test_batch_size)

            dataloaders.update({task_name:
                                    {"train": train_dataloader,
                                     "validation_matched": eval_dataloader_matched,
                                     "validation_mismatched": eval_dataloader_mismatched,
                                     "test_matched": test_dataloader_matched,
                                     "test_mismatched": test_dataloader_mismatched}})
        else:
            train_dataloader = DataLoader(dataset=task["train"],
                                          shuffle=True,
                                          collate_fn=default_data_collator,
                                          batch_size=args.train_batch_size)
            eval_dataloader = DataLoader(dataset=task["validation"],
                                          shuffle=False,
                                          collate_fn=default_data_collator,
                                          batch_size=args.test_batch_size)
            test_dataloader = DataLoader(dataset=task["test"],
                                          shuffle=False,
                                          collate_fn=default_data_collator,
                                          batch_size=args.test_batch_size)

            dataloaders.update({task_name:
                                    {"train": train_dataloader,
                                     "validation": eval_dataloader,
                                     "test": test_dataloader}})


    '''
    if args.resume_from_checkpoint:
        pass
    '''

    ## Pre-Train ##
    pre_train_eval_loop_glue(models, dataloaders, tokenizer, device, args, use_wandb=False)




    '''
     if args.wandb:
        pre_result_logs = wandb.Table(dataframe=pre_result)
        wandb.log({"pre_result": pre_result_logs})
    ## Prune ##
    for (name, model), (_, dataloader_dict) in zip(models, dataloaders):
        prune_loader = dataloader_dict["train"]
        generator.count_prunable_parameters(model)
        print('Pruning for {} epochs.'.format(args.prune_epochs))
        sparsity = 10 ** (-float(args.compression))
        prune_result = prune_loop(model, args.pruner, prune_loader, None, device, sparsity,
                                  args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize,
                                  args.prune_train_mode, args.shuffle, args.invert, use_wandb=args.wandb)
        if args.wandb:
            prune_result_logs = wandb.Table(dataframe=prune_result)
            wandb.log({"prune_result": prune_result_logs})

        if args.save:
            prune_result.to_pickle(f"{args.result_dir}/compression_{name}.pkl")

    ## Post-Train ##
    post_result = post_train_eval_loop_glue(models, dataloaders, device, args, use_wandb=False)      
    if args.wandb:
        post_result_logs = wandb.Table(dataframe=post_result)
        wandb.log({"post_result": post_result_logs})
    '''
    ## Display Results ##
    if args.wandb:
        wandb.finish()