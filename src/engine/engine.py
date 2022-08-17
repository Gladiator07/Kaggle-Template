# Inspired from HuggingFace Trainer and Tez
# https://huggingface.co/docs/transformers/main_classes/trainer
# https://github.com/abhishekkrthakur/tez

import math
import multiprocessing
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainerArguments:
    per_device_train_batch_size: Optional[int] = 32
    per_device_valid_batch_size: Optional[int] = 32

    num_train_epochs: Optional[int] = 5
    gradient_accumulation_steps: Optional[int] = 1
    clip_grad_norm: Optional[float] = None
    num_workers: Optional[int] = -1
    mixed_precision: Optional[str] = None  # bf16, fp16

    # dataloader args
    train_shuffle: Optional[bool] = True
    valid_shuffle: Optional[bool] = False
    train_drop_last: Optional[bool] = False
    valid_drop_last: Optional[bool] = False
    test_drop_last: Optional[bool] = False
    test_shuffle: Optional[bool] = False
    pin_memory: Optional[bool] = True

    # scheduler args
    # TODO: add support for a custom scheduler and stepping after epoch or batch option
    scheduler_type: Optional[str] = "cosine"  # linear
    num_warmup_steps: Optional[Union[float, int]] = 0.1

    # misc
    log_to_wandb: Optional[bool] = False
    log_every_n_steps: Optional[int] = 10
    log_mode: Optional[bool] = "print"  # log
    save_best_checkpoint: Optional[bool] = True
    save_last_checkpoint: Optional[bool] = True


"""
A boiler plate Trainer class which can be used with minimal to no customization for most of the problems
For now no plan to make it fully generalizable, just copy paste this file and modify according to the project/competition if required
"""


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,  # passing from outside for now to have the flexibility to modify param groups
        args: TrainerArguments,
        train_dataset: Optional[Dataset] = None,
        valid_dataset: Optional[Dataset] = None,
        train_sampler: Optional[Sampler] = None,
        valid_sampler: Optional[Sampler] = None,
        train_collate_fn: Optional[Callable] = None,
        valid_collate_fn: Optional[Callable] = None,
        train_dataloader: Optional[DataLoader] = None,
        valid_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        - Either pass train_dataset, valid_dataset and trainer will create dataloader on its own.
        - If you want special samplers and collate functions then you can pass them too and they will be assigned to dataloader.
        - Or you can simply pass the dataloaders and this trainer will use those without constructing on its own.
        """

        # model should be instantiated before accelerator init
        # as on TPU it may load this model on all processes
        # and RAM may explode!
        self.model = model
        self.optimizer = optimizer
        self.args = args

        self._init_accelerator()

        # dataset
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # sampler
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        # collate fns
        self.train_collate_fn = train_collate_fn
        self.valid_collate_fn = valid_collate_fn
        self.compute_metrics = compute_metrics
        # dataloader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        if self.args.num_workers == -1:
            self.args.num_workers = multiprocessing.cpu_count()
            if self.args.num_workers > 4:
                self.args.num_workers -= 2

        # if not provided then create dataloaders from datasets/samplers/collate_fns
        if self.train_dataloader is None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=self.args.train_shuffle,
                collate_fn=self.train_collate_fn,
                sampler=self.train_sampler,
                num_workers=self.args.num_workers,
                drop_last=self.args.train_drop_last,
            )

        if self.valid_dataloader is None:
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=self.args.per_device_valid_batch_size,
                shuffle=self.args.valid_shuffle,
                collate_fn=self.valid_collate_fn,
                sampler=self.valid_sampler,
                num_workers=self.args.num_workers,
                drop_last=self.args.valid_drop_last,
            )

        # math around schedulers
        if isinstance(self.args.num_warmup_steps, float):
            num_warmup_steps = self.args.num_warmup_steps * len(self.train_dataloader)
        elif isinstance(self.args.num_warmup_steps, int):
            num_warmup_steps = self.args.num_warmup_steps
        num_warmup_steps = math.ceil(num_warmup_steps)
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )

        self.num_train_steps = num_update_steps_per_epoch * self.args.num_train_epochs

        if self.args.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_train_steps,
            )

        elif self.args.scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_train_steps,
            )
            # prepare for distributed training aka put everything in HF Accelerate
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.valid_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.valid_dataloader,
                self.lr_scheduler,
            )

            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) / self.args.gradient_accumulation_steps
            )
            self.num_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            # Afterwards we recalculate our number of training epochs
            self.args.num_train_epochs = math.ceil(
                self.num_train_steps / num_update_steps_per_epoch
            )

    def _init_accelerator(self, *args, **kwargs):
        self.accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=self.args.mixed_precision
            if self.args.mixed_precision is not None
            else "no",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def _init_global_progress_bar(self):
        self.global_prog_bar = tqdm(
            range(self.args.num_train_epochs * len(self.train_dataloader)),
            disable=not self.accelerator.is_main_process,
        )

    def train_one_step(self, batch):
        with self.accelerator.accumulate(self.model):

            _, loss = self.model(**batch)
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            self.global_prog_bar.set_postfix(loss=loss.item())
            self.global_prog_bar.update(1)

        return loss

    def train_one_epoch(self, dataloader):
        total_loss = 0.0
        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_one_step(batch)
            total_loss += loss.detach().float()

        avg_loss = total_loss.item() / len(dataloader)
        return avg_loss

    @torch.no_grad()
    def evaluate_one_step(self, batch):
        model_out, loss = self.model(**batch)
        labels = batch["label"]
        return model_out, loss, labels

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Model forward method (from the model class) can return anything you want in the dictionary format or just a single output vector
        The reason for supporting the dictionary format is if any advance processing of the multiple outputs is required by
        the `compute_metrics` for calculation of metrics.
        """
        # TODO: add support for doing only evaluation by just loading the model state
        all_model_outs = []
        all_labels = []
        print("In evaluation loop")
        total_loss = 0.0
        eval_pbar = tqdm(
            range(len(dataloader)),
            disable=self.accelerator.is_main_process,
            leave=False,
        )
        self.model.eval()
        for batch_idx, batch in enumerate(dataloader):
            model_out, loss, labels = self.evaluate_one_step(batch)

            model_out = self.accelerator.gather_for_metrics(model_out)
            all_model_outs.append(model_out.cpu().numpy())
            loss, labels = self.accelerator.gather_for_metrics((loss, labels))
            total_loss += loss

            all_labels.append(labels.cpu().numpy())
            eval_pbar.update(1)
        eval_pbar.close()
        avg_loss = total_loss / len(dataloader)
        all_model_outs = np.concatenate(all_model_outs)
        all_labels = np.concatenate(all_labels)
        metrics = self.compute_metrics(all_model_outs, all_labels)
        return all_model_outs, all_labels, metrics

    def save_model(self, weights_only: Optional[bool] = False, **kwargs):
        # make sure to handle distributed case here
        pass

    def load_model(self, weights_only: Optional[bool] = False, **kwargs):
        # make sure to handle distributed case here
        pass

    def predict(self, **kwargs):
        # make sure to handle distributed case here
        pass

    def fit(self):
        self._train_startup_log()
        self._init_global_progress_bar()
        for _ in range(self.args.num_train_epochs):
            trn_epoch_loss = self.train_one_epoch(self.train_dataloader)
            val_outs = self.evaluate(self.valid_dataloader)

    def _train_startup_log(self):
        total_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Num examples = {len(self.train_dataset)}")
        self.accelerator.print(f"  Num Epochs = {self.args.num_train_epochs}")
        self.accelerator.print(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        self.accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        self.accelerator.print(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        self.accelerator.print(f"  Total optimization steps = {self.num_train_steps}")
