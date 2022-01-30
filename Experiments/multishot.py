import torch
from Models.pl_model import PLModel, CustomPruningCallback, ImpConservationCallback, ScheduleConservationCallback
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import GPUtil
import wandb

def run(args):
    ## Unpack arguments ##
    train_args = args['train']
    model_args = args['model']
    prune_args = args['pruning']
    args = args['general']

    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    pl.seed_everything(args.seed)

    ## Model ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = PLModel(**model_args)

    ## Logger ##
    if args.log:
        wandb_logger = WandbLogger(project='vit_synflow')
        train_args.update({'logger': wandb_logger})

    ## Callbacks ##
    callbacks = []
    if args.save_checkpoint:
        custom_checkpoint_callback = ModelCheckpoint(dirpath=args.result_dir,
                                                                  filename=args.exp_id,
                                                                  save_last=True)
        callbacks.append(custom_checkpoint_callback)
    if args.prune:
        pruning_callback = CustomPruningCallback(model,
                                                 prune_args)
        callbacks.append(pruning_callback)

    if args.imp_conservation:
        imp_callback = ImpConservationCallback(model)
        callbacks.append(imp_callback)

    if args.schedule_conservation:
        schedule_callback = ScheduleConservationCallback(model, args.compression_list, args.prune_epoch_list, prune_args.prune_bias, prune_args.prune_batchnorm, prune_args.prune_residual)

    ## Instantiate Model and Trainer
    trainer = pl.Trainer(callbacks=callbacks, **train_args)

    ## Save Original ##
    torch.save(model.model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(model.optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(model.scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Train-Prune Loop ##
    for compression in args.compression_list:
        for level in args.level_list:
            print('{} compression ratio, {} train-prune levels'.format(compression, level))
            
            # Reset Model, Optimizer, and Scheduler
            model.model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=model.device))
            model.optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=model.device))
            model.scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=model.device))
            
            for l in range(level):

                trainer.fit(model)
                GPUtil.showUtilization()

                # Reset Model's Weights
                original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=model.device)
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                model.optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=model.device))
                model.scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=model.device))

    if args.log:
        wandb.finish()

