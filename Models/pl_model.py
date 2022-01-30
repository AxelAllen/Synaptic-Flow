import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import pandas as pd
import numpy as np
from tqdm import tqdm

import Utils.load
from sam.sam import SAM
from Utils import generator, load
from pthflops import count_ops
from torchvision import transforms, datasets
from Utils import custom_datasets
from torch.utils.data import random_split
import math
from Layers import layers

class PLModel(pl.LightningModule):
    def __init__(self, **hparams):
        super(PLModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.input_shape, self.num_classes = load.dimension(self.hparams.dataset)
        if self.hparams.model_class == 'transformer':
            self.model = load.model(self.hparams.model, self.hparams.model_class).load_model(self.hparams.model,
                                                                                        self.input_shape,
                                                                                        self.hparams.patch_size,
                                                                                        self.hparams.emb_dim,
                                                                                        self.hparams.mlp_dim,
                                                                                        self.hparams.num_heads,
                                                                                        self.hparams.num_layers,
                                                                                        self.num_classes,
                                                                                        self.hparams.pretrained)
        else:
            self.model = load.model(self.hparams.model, self.hparams.model_class)(self.input_shape,
                                                                                  self.num_classes,
                                                                                  self.hparams.dense_classifier,
                                                                                  self.hparams.pretrained)
        self.automatic_optimization = False


    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        data, target = batch
        output = self.model(data)
        loss = self.loss(output, target)
        acc1 = accuracy(preds=output, target=target, top_k=1)
        acc5 = accuracy(preds=output, target=target, top_k=5)
        return loss, acc1, acc5


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        train_loss, acc1, acc5 = self._step(batch)
        if isinstance(opt, SAM):
            train_loss = train_loss.mean()
        opt.zero_grad()

        self.log('train/loss', train_loss)
        self.log('train/acc1', acc1)
        self.log('train/acc5', acc5)

        self.manual_backward(train_loss)

        def closure():
            data, target = batch
            output = self.model(data)
            loss = self.loss(output, target).mean()
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        if isinstance(opt, SAM):
            opt.step(closure=closure)
        else:
            opt.step(closure=None)

        return {'loss': train_loss, 'acc1': acc1, 'acc5': acc5}

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        val_loss, acc1, acc5 = self._step(batch)

        self.log('val/loss', val_loss, on_step=False, on_epoch=True)
        self.log('val/acc1', acc1, on_step=False, on_epoch=True)
        self.log('val/acc5', acc5, on_step=False, on_epoch=True)

        return {'loss': val_loss, 'acc1': acc1, 'acc5': acc5}

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x['loss'].detach.numpy() for x in outputs])
        avg_acc1 = np.mean([x['acc1'].detach.numpy() for x in outputs])
        avg_acc5 = np.mean([x['acc5'].detach.numpy() for x in outputs])

        return {'avg_loss': avg_loss, 'avg_acc1': avg_acc1, 'avg_acc5': avg_acc5}

    def test_step(self, batch, batch_idx):
        test_loss, acc1, acc5 = self._step(batch)

        self.log('test/loss', test_loss, on_step=False,  on_epoch=True)
        self.log('test/acc1', acc1, on_step=False, on_epoch=True)
        self.log('test/acc5', acc5, on_step=False, on_epoch=True)

        return {'loss': test_loss, 'acc1': acc1, 'acc5': acc5}

    def test_epoch_end(self, outputs):
        avg_loss = np.mean([x['loss'] for x in outputs])
        avg_acc1 = np.mean([x['acc1'] for x in outputs])
        avg_acc5 = np.mean([x['acc5'] for x in outputs])

        return {'avg_loss': avg_loss, 'test_acc1': avg_acc1, 'test_acc5': avg_acc5}

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def configure_optimizers(self):
        opt_class, opt_kwargs = load.optimizer(self.hparams.optimizer)
        if self.hparams.sam:
            print(f"Sharpness Aware Minimization enabled.")
            opt_kwargs.update({'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay})
            optimizer = SAM(generator.parameters(self.model), opt_class, **opt_kwargs)
        else:
            print(f"Sharpness Aware Minimization disabled. Using base optimizer {opt_class} instead.")
            optimizer = opt_class(generator.parameters(self.model), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_drops,
                                             gamma=self.hparams.lr_drop_rate)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def get_transform(self, size, padding, mean, std, preprocess):
        transform = []
        if preprocess:
            transform.append(transforms.RandomCrop(size=size, padding=padding))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        return transforms.Compose(transform)

    def load_data(self, dataset, train):
        # Dataset
        if dataset == 'mnist':
            mean, std = (0.1307,), (0.3081,)
            transform = self.get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
            dataset = datasets.MNIST('Data', train=train, download=False, transform=transform)
        if dataset == 'cifar10':
            mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
            transform = self.get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
            dataset = datasets.CIFAR10('Data', train=train, download=False, transform=transform)
        if dataset == 'cifar100':
            mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
            transform = self.get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
            dataset = datasets.CIFAR100('Data', train=train, download=False, transform=transform)
        if dataset == 'tiny-imagenet':
            mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
            transform = self.get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
            dataset = custom_datasets.TINYIMAGENET('Data', train=train, download=False, transform=transform)
        if dataset == 'imagenet':
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
            folder = 'Data/imagenet_raw/{}'.format('train' if train else 'val')
            dataset = datasets.ImageFolder(folder, transform=transform)

        return dataset

    def dataloader(self, dataset, batch_size, train, workers, length=None):

        # Dataloader
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
        shuffle = train is True
        if length is not None:
            indices = torch.randperm(len(dataset))[:length]
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 **kwargs)

        return dataloader

    def prepare_data(self):
        dataset = self.hparams.dataset
        if dataset == 'mnist':
            datasets.MNIST('Data', train=True, download=True)
            datasets.MNIST('Data', train=False, download=True)
        if dataset == 'cifar10':
            datasets.CIFAR10('Data', train=True, download=True)
            datasets.CIFAR10('Data', train=False, download=True)
        if dataset == 'cifar100':
            datasets.CIFAR100('Data', train=True, download=True)
            datasets.CIFAR100('Data', train=False, download=True)
        if dataset == 'tiny-imagenet':
            custom_datasets.TINYIMAGENET('Data', train=True, download=True)
            custom_datasets.TINYIMAGENET('Data', train=False, download=True)

    def setup(self, stage=None):
        train_data = self.load_data(self.hparams.dataset, train=True)
        train_len = math.floor(0.8*len(train_data))
        val_len = len(train_data)-train_len
        train_data, val_data = random_split(train_data, [train_len, val_len])
        test_data = self.load_data(self.hparams.dataset, train=False)

        self.train_loader, self.val_loader, self.test_loader, self.prune_loader = \
            self.dataloader(train_data, self.hparams.batch_size, True, self.hparams.num_workers),\
            self.dataloader(val_data, self.hparams.batch_size, False,  self.hparams.num_workers),\
            self.dataloader(test_data, self.hparams.batch_size, False,  self.hparams.num_workers),\
            self.dataloader(train_data, self.hparams.batch_size, False, self.hparams.num_workers, length=self.hparams.prune_dataset_ratio * self.num_classes)
        self.loss = nn.CrossEntropyLoss()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def prune_dataloader(self):
        return self.prune_loader

class CustomPruningCallback(pl.Callback):

    def __init__(self, module, args):
        self.params = args
        self.pruner = self.load_pruner(args.prune_method)
        self.sparsity = 10 ** (-float(args.compression))
        self.schedule = args.schedule
        self.scope = args.scope
        self.epochs = args.epochs
        self.module = module
        self.names, self.inv_size = self.layer_names(module.model)
        self.average_scores = []
        self.unit_scores = []
        self.layer_conservation = args.layer_conservation
        self.unit_conservation = args.unit_conservation

    def on_train_start(self, trainer, pl_module):
        if pl_module.hparams.pre_epochs == 0:
            self.prune_loop(pl_module, self.pruner)
        if self.layer_conservation:
            average_score = self.average_layer_score(pl_module.model, self.pruner.scores)
            self.average_scores.append(average_score)
        if self.unit_conservation:
            unit_score = self.unit_score_sum(pl_module.model, self.pruner.scores)
            self.unit_scores.append(unit_score)

    def on_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch == pl_module.hparams.pre_epochs-1:
            self.prune_loop(pl_module, self.pruner)
        if self.layer_conservation:
            average_score = self.average_layer_score(pl_module.model, self.pruner.scores)
            self.average_scores.append(average_score)
        if self.unit_conservation:
            unit_score = self.unit_score_sum(pl_module.model, self.pruner.scores)
            self.unit_scores.append(unit_score)


    def on_train_end(self, trainer, pl_module):
        model = pl_module.model
        ops = self.count_flops(model)
        summary = self.summary(self.pruner.scores,
                               lambda p: generator.prunable(p, self.params['prune_batchnorm'], self.params['prune_residual']))

        if self.layer_conservation:
            np.save('{}/{}'.format(trainer.default_root_dir, str(type(self.pruner))), np.array(self.average_scores))
            np.save('{}/{}'.format(trainer.default_root_dir, 'inv-size'), self.inv_size)
        if self.unit_conservation:
            np.save('{}/{}'.format(trainer.default_root_dir, str(type(self.pruner))), self.unit_scores)
        return {'flops': ops, 'summary': summary}

    def load_pruner(self, prune_method):
        return Utils.load.pruner(prune_method)

    def prune_loop(self, model, pruner, reinitialize=False, shuffle=False, invert=False):
        r"""Applies score mask loop iteratively to a final sparsity level.
        """
        model.eval()
        loss = model.loss
        dataloader = model.prune_loader
        device = model.device


        # Prune model
        for epoch in tqdm(range(self.epochs)):
            pruner.score(model.model, loss, dataloader, device)
            if self.schedule == 'exponential':
                sparse = self.sparsity ** ((epoch + 1) / self.epochs)
            elif self.schedule == 'linear':
                sparse = 1.0 - (1.0 - self.sparsity) * ((epoch + 1) / self.epochs)
            # Invert scores
            if invert:
                pruner.invert()
            pruner.mask(sparse, self.scope)

        # Reainitialize weights
        if reinitialize:
            model._initialize_weights()

        # Shuffle masks
        if shuffle:
            pruner.shuffle()

        # Confirm sparsity level
        remaining_params, total_params = pruner.stats()
        if np.abs(remaining_params - total_params * self.sparsity) >= 5:
            print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params,
                                                                                total_params * self.sparsity))
            quit()

    def count_flops(self, model):
        ## Count Flops ##
        (data, _) = next(iter(model.train_loader))
        ops, _ = count_ops(model, data)
        model.log('flops', ops)
        return ops

    def summary(self, scores, prunable):
        r"""Summary of compression results for a model.
        """
        rows = []
        for name, module in self.model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                pruned = prunable(module) and id(param) in scores.keys()
                if pruned:
                    sparsity = getattr(module, pname + '_mask').detach().cpu().numpy().mean()
                    score = scores[id(param)].detach().cpu().numpy()
                else:
                    sparsity = 1.0
                    score = np.zeros(1)
                shape = param.detach().cpu().numpy().shape
                score_mean = score.mean()
                score_var = score.var()
                score_sum = score.sum()
                score_abs_mean = np.abs(score).mean()
                score_abs_var = np.abs(score).var()
                score_abs_sum = np.abs(score).sum()
                rows.append([name, pname, sparsity, np.prod(shape), shape,
                             score_mean, score_var, score_sum,
                             score_abs_mean, score_abs_var, score_abs_sum,
                             pruned])
        columns = ['module', 'param', 'sparsity', 'size', 'shape', 'score mean', 'score variance',
                   'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
        return pd.DataFrame(rows, columns=columns)

    ## Compute Layer Name and Inv Size ##
    def layer_names(self, model):
        names = []
        inv_size = []
        for name, module in model.named_modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                num_elements = np.prod(module.weight.shape)
                if module.bias is not None:
                    num_elements += np.prod(module.bias.shape)
                names.append(name)
                inv_size.append(1.0 / num_elements)
        return names, inv_size

    ## Compute Average Layer Score ##
    def average_layer_score(self, model, scores):
        average_scores = []
        for name, module in model.named_modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                W = module.weight
                W_score = scores[id(W)].detach().cpu().numpy()
                score_sum = W_score.sum()
                num_elements = np.prod(W.shape)

                if module.bias is not None:
                    b = module.bias
                    b_score = scores[id(b)].detach().cpu().numpy()
                    score_sum += b_score.sum()
                    num_elements += np.prod(b.shape)

                average_scores.append(np.abs(score_sum / num_elements))
        return average_scores

        ## Compute per Neuron Score ##

    def unit_score_sum(self, model, scores, compute_linear=False):
        in_scores = []
        out_scores = []
        for name, module in model.named_modules():
            if (isinstance(module, layers.Linear) and compute_linear):
                W = module.weight
                b = module.bias

                W_score = scores[id(W)].detach().cpu().numpy()
                b_score = scores[id(b)].detach().cpu().numpy()

                in_scores.append(W_score.sum(axis=1) + b_score)
                out_scores.append(W_score.sum(axis=0))
            if isinstance(module, layers.Conv2d):
                W = module.weight
                W_score = scores[id(W)].detach().cpu().numpy()
                in_score = W_score.sum(axis=(1, 2, 3))
                out_score = W_score.sum(axis=(0, 2, 3))

                if module.bias is not None:
                    b = module.bias
                    b_score = scores[id(b)].detach().cpu().numpy()
                    in_score += b_score

                in_scores.append(in_score)
                out_scores.append(out_score)

        in_scores = np.concatenate(in_scores[:-1])
        out_scores = np.concatenate(out_scores[1:])
        return in_scores, out_scores

class ImpConservationCallback(pl.Callback):

    def __init__(self, module):
        self.module = module
        self.Wscore = []
        self.inv_size = self.layer_names(module.model)

    def layer_names(self, model):
        names = []
        inv_size = []
        for name, module in model.named_modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                num_elements = np.prod(module.weight.shape)
                if module.bias is not None:
                    num_elements += np.prod(module.bias.shape)
                names.append(name)
                inv_size.append(1.0/num_elements)
        return inv_size

    ## Compute Average Mag Score ##
    def average_mag_score(self, model):
        average_scores = []
        for module in model.modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                W = module.weight.detach().cpu().numpy()
                W_score = W**2
                score_sum = W_score.sum()
                num_elements = np.prod(W.shape)

                if module.bias is not None:
                    b = module.bias.detach().cpu().numpy()
                    b_score = b**2
                    score_sum += b_score.sum()
                    num_elements += np.prod(b.shape)

                average_scores.append(np.abs(score_sum / num_elements))
        return average_scores

    def imp_conservation(self, model):
        self.Wscore.append(self.average_mag_score(model))

    def on_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch >= trainer.max_epochs - pl_module.hparams.post_epochs:
            self.imp_conservation(pl_module.model)

    def on_train_end(self, trainer, pl_module):
        np.save('{}/{}'.format(trainer.default_root_dir, 'inv-size'), self.inv_size)
        np.save('{}/score'.format(trainer.default_root_dir), np.array(self.Wscore))




class ScheduleConservationCallback(pl.Callback):

    def __init__(self, pl_module, compression_list, prune_epoch_list, prune_bias=False, prune_batchnorm=False, prune_residual=False):
        self.results = []
        self.module = pl_module
        self.model = pl_module.model
        self.compression_list = compression_list
        self.prune_epoch_list = prune_epoch_list
        self.prune_bias = prune_bias
        self.prune_batchnorm = prune_batchnorm
        self.prune_residual = prune_residual

    def on_train_start(self, trainer, pl_module):
        self.schedule_consevation(trainer)

    def on_train_end(self, trainer, pl_module):
        np.save('{}/ratios'.format(trainer.default_root_dir), np.array(self.results))

    def score(self, parameters, model, loss, dataloader):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(self.module.device)
        output = model(input)
        maxflow = torch.sum(output)
        maxflow.backward()
        scores = {}
        for _, p in parameters:
            scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()
        nonlinearize(model, signs)

        return scores, maxflow.item()

    def mask(self, parameters, scores, sparsity):
        global_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        cutsize = 0
        if not k < 1:
            cutsize = torch.sum(torch.topk(global_scores, k, largest=False).values).item()
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in parameters:
                score = scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
        return cutsize

    @torch.no_grad()
    def apply_mask(self, parameters):
        for mask, param in parameters:
            param.mul_(mask)

    def schedule_consevation(self, trainer):
        for style in ['linear', 'exponential']:
            print(style)
            sparsity_ratios = []
            for i, exp in enumerate(self.compression_list):
                max_ratios = []
                for j, epochs in enumerate(self.prune_epoch_list):
                    self.model.load_state_dict(torch.load("{}/model.pt".format(trainer.default_root_dir), map_location=self.module.device))
                    parameters = list(
                        generator.masked_parameters(self.model, self.prune_bias, self.prune_batchnorm, self.prune_residual))
                    self.model.eval()
                    ratios = []
                    for epoch in tqdm(range(epochs)):
                        self.apply_mask(parameters)
                        scores, maxflow = self.score(parameters, self.model, self.module.loss, self.module.prune_loader, self.module.device)
                        sparsity = 10 ** (-float(exp))
                        if style == 'linear':
                            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
                        if style == 'exponential':
                            sparse = sparsity ** ((epoch + 1) / epochs)
                        cutsize = self.mask(parameters, scores, sparse)
                        ratios.append(cutsize / maxflow)
                    max_ratios.append(max(ratios))
                sparsity_ratios.append(max_ratios)
            self.results.append(sparsity_ratios)


