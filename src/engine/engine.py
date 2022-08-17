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

"""
TODO:
[ ] Print evaluation metrics nicely after each epoch
[ ] Add clip_grad_norm to training loop
[ ] Add Weights & Biases logging (handle with care in distributed settings
[ ] Add saving and loading of model checkpoint (handle with care in distributed settings)
[ ] Add option to choose between print to console or log to a file
[ ] Add a `final_summary` method which will print the complete summary per epoch (losses and metrics) in a table form
[ ] Test the code thoroughly with multi-GPU setup
[ ] Test the code thoroughly with TPU setup 
[ ] Add helpful print/log messages while training
[ ] Maybe wrap up the math around scheduler in a private method
[ ] Write a predict function with a support to easily add TTA if required
[ ] Add log_every_n_steps functionality (maybe to reduce bottlenecks while on TPUs)
[ ] Add docstrings 😛
"""


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

        self.model = model
        self.optimizer = optimizer
        self.args = args

        # internals
        self._current_epoch = 1
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
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        # total training/optimization steps
        self.num_train_steps = (
            self.args.num_train_epochs * self.num_update_steps_per_epoch
        )

        # total warmup steps
        if isinstance(self.args.num_warmup_steps, float):
            self.args.num_warmup_steps = math.ceil(
                self.args.num_warmup_steps * self.num_train_steps
            )
        # define linear/cosine scheduler
        if self.args.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.num_train_steps,
            )

        elif self.args.scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.num_train_steps,
            )

        # prepare for distributed training aka put everything in 🤗 Accelerate 🚀
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

        # re-calculate total training steps and epoch as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        self.num_train_steps = (
            self.args.num_train_epochs * self.num_update_steps_per_epoch
        )
        self.args.num_train_epochs = math.ceil(
            self.num_train_steps / self.num_update_steps_per_epoch
        )

        self.total_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.is_main_process
            * self.args.gradient_accumulation_steps
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

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch and return the average loss
        Note: the model must return the loss from its forward function
        """
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            with self.accelerator.accumulate(self.model):
                logits, loss = self.model(**batch)
                total_loss += loss.detach().float()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                self.global_prog_bar.set_postfix(loss=loss.item())
                self.global_prog_bar.update(1)

        # can also calculate metrics here for training epoch via `compute_metrics` Calllable function
        # ...
        return total_loss.item() / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        all_logits = []
        all_labels = []
        total_loss = 0
        eval_pbar = tqdm(
            range(len(dataloader)),
            disable=not self.accelerator.is_main_process,
            leave=False,
        )
        self.model.eval()
        for step, batch in enumerate(dataloader):
            logits, loss = self.model(**batch)
            total_loss += loss.item()
            all_logits.append(self.accelerator.gather_for_metrics(logits).cpu().numpy())
            all_labels.append(
                self.accelerator.gather_for_metrics(batch["label"].cpu().numpy())
            )
            eval_pbar.update(1)
        eval_pbar.close()
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        eval_metrics = self.compute_metrics(all_logits, all_labels)

        return all_logits, all_labels, eval_metrics, total_loss

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
        self._train_startup_log_msg()
        self._init_global_progress_bar()
        for epoch in range(self.args.num_train_epochs):
            trn_epoch_loss = self.train_one_epoch(self.train_dataloader)
            logits, labels, eval_metrics, val_epoch_loss = self.evaluate(
                self.valid_dataloader
            )
            self._log_epoch_summary()
            summary_metrics = []
            for m, v in eval_metrics.items():
                tmp_str = f"valid_{m}: {v:.4f}"
                summary_metrics.append(tmp_str)
            eval_metrics = "  |  ".join(summary_metrics)
            summary_str = f"  Epoch {self._current_epoch}  |  train_loss: {trn_epoch_loss:.4f}  |  valid_loss: {val_epoch_loss:.4f}  |  {eval_metrics}"
            self.accelerator.print(summary_str)
            self._current_epoch += 1

    def _train_startup_log_msg(self):
        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Num examples = {len(self.train_dataset)}")
        self.accelerator.print(f"  Num Epochs = {self.args.num_train_epochs}")
        self.accelerator.print(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        self.accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}"
        )
        self.accelerator.print(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        self.accelerator.print(f"  Total optimization steps = {self.num_train_steps}")

    def _log_epoch_summary(self):
        pass
