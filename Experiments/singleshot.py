import torch
from Models.pl_model import PLModel, CustomPruningCallback, ImpConservationCallback, ScheduleConservationCallback
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import GPUtil
import wandb

# from pthflops import count_ops

def run(args):

    ## Unpack arguments ##
    train_args = args['training']
    model_args = args['model']
    prune_args = args['pruning']
    args = args['general']


    ## Random Seed and Device ##
    pl.seed_everything(args.seed)

    ## Model ##
    print('Creating {}-{} model.'.format(model_args['model_class'], model_args['model']))
    model = PLModel(**model_args)

    ## Logger ##
    wandb_logger = WandbLogger(project='vit_synflow')

    ## Callbacks ##
    callbacks = []
    if args.save_checkpoint:
        custom_checkpoint_callback = ModelCheckpoint(dirpath=args.result_dir,
                                                                filename=args.expid,
                                                                save_last=True)
        callbacks.append(custom_checkpoint_callback)
    if args.prune:
        pruning_callback = CustomPruningCallback(model, prune_args)
        callbacks.append(pruning_callback)

    if args.imp_conservation:
        imp_callback = ImpConservationCallback(model)
        callbacks.append(imp_callback)

    train_args.update({'max_epochs': model_args['pre_epochs'] + model_args['post_epochs']})
     

    ## Instantiate Model and Trainer
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **train_args)

    ## Train and Prune ##
    trainer.fit(model)
    #GPUtil.showUtilization()

    ## Test ##

    trainer.test(model, ckpt_path=None)
    #GPUtil.showUtilization()

    wandb.finish()

    





