# Inspired from HuggingFace Trainer and Tez
# https://huggingface.co/docs/transformers/main_classes/trainer
# https://github.com/abhishekkrthakur/tez
import math
import multiprocessing
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from utils import AverageMeter

"""
TODO:
[x] Print valuation metrics nicely after each epoch
[x] Add clip_grad_norm to training loop
[x] Make sure that loss calculated is correct
[ ] Maybe pass even the dataloaders to the class to keep the class simple and less cluttered
[ ] Add Weights & Biases logging (handle with care in distributed settings
[ ] Handle tie and untie model weights in case of TPU
[x] Add saving and loading of model checkpoint (handle with care in distributed settings)
[ ] Check optimization stuff (total training steps, accumulation , lr scheduler)
[ ] Add a predict method which can perform prediction from loading a model state. Example: `trainer.predict(test_dataset, **kwargs)`
[ ] Write a predict function with a support to easily add TTA if required
[ ] Add option to choose between print to console or log to a file
[ ] Add a `final_summary` method which will print the complete summary per epoch (losses and metrics) in a table form
[ ] Add helpful print/log messages while training
[ ] Maybe wrap up the math around scheduler in a private method
[ ] Add log_every_n_steps functionality (maybe to reduce bottlenecks while on TPUs)
[ ] Test the code thoroughly with multi-GPU setup
[ ] Test the code thoroughly with TPU setup 
[ ] Add type hints
[ ] Add docstrings 😛
"""


@dataclass
class EvalOutputs:
    """
    A dataclass to store outputs from `Trainer.evaluate` method
    Change this class and the method if you want to return something else
    """

    logits: np.ndarray
    labels: np.ndarray
    metrics: Dict[str, float]


@dataclass
class TrainerArguments:
    output_dir: str
    per_device_train_batch_size: Optional[int] = 32
    per_device_val_batch_size: Optional[int] = 32

    num_train_epochs: Optional[int] = 5
    gradient_accumulation_steps: Optional[int] = 1
    max_grad_norm: Optional[float] = 1.0
    num_workers: Optional[int] = -1
    mixed_precision: Optional[str] = None  # bf16, fp16

    # dataloader args
    train_shuffle: Optional[bool] = True
    val_shuffle: Optional[bool] = False
    train_drop_last: Optional[bool] = False
    val_drop_last: Optional[bool] = False
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
    save_weights_only: Optional[bool] = True
    metric_for_best_model: Optional[str] = "accuracy"


"""
A boiler plate Trainer class which can be used with minimal to no code changes for most of the problems
For now no plan to make it fully generalizable, just copy paste this file and modify according to the project/competition if required
"""


@dataclass
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,  # model is required
        args: TrainerArguments = None,  # args is required
        optimizer: torch.optim = None,  # passing from outside for now to have the flexibility to modify param groups
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
        train_collate_fn: Optional[Callable] = None,
        val_collate_fn: Optional[Callable] = None,
        test_collate_fn: Optional[Callable] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        - Either pass train_dataset, val_dataset and trainer will create dataloader on its own.
        - If you want special samplers and collate functions then you can pass them too and they will be assigned to dataloader.
        - Or you can simply pass the dataloaders and this trainer will use those without constructing on its own.
        """
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.compute_metrics = compute_metrics
        # internals
        self._trn_loss_meter = AverageMeter("train_loss", ":.4e")
        self._val_loss_meter = AverageMeter("val_loss", ":.4e")
        self._current_epoch = 0
        self._current_epoch_train_loss = 0.0
        self._current_epoch_val_loss = 0.0
        self._current_val_metrics = {}
        self._best_val_metric = 0
        self._accelerator = None

    def _init_accelerator(self, *args, **kwargs):
        """
        Initialize the 🤗 Accelerator
        """
        self._accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=self.args.mixed_precision if self.args.mixed_precision is not None else "no",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def _set_num_workers(self):
        if self.args.num_workers == -1:
            self.args.num_workers = multiprocessing.cpu_count()
            if self.args.num_workers > 4:
                self.args.num_workers -= 2

    def _init_trainer(self):
        """
        Initialize trainer for training and evaluation routines
        """
        self._set_num_workers()
        self._init_accelerator()
        # if not provided then create dataloaders from datasets/samplers/collate_fns
        if self.train_dataloader is None and self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=self.args.train_shuffle,
                collate_fn=self.train_collate_fn,
                sampler=self.train_sampler,
                num_workers=self.args.num_workers,
                drop_last=self.args.train_drop_last,
            )

        if self.val_dataloader is None and self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.args.per_device_val_batch_size,
                shuffle=self.args.val_shuffle,
                collate_fn=self.val_collate_fn,
                sampler=self.val_sampler,
                num_workers=self.args.num_workers,
                drop_last=self.args.val_drop_last,
            )
        # prepare for distributed training aka put everything in 🤗 Accelerate 🚀
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        ) = self._accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.val_dataloader)

        # TODO: handle weight tying of model after pushed to XLA device here
        # https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks

        if self.train_dataloader is not None:
            # re-calculate total training steps and epoch as the size of the training dataloader may have changed.
            self.num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) / self.args.gradient_accumulation_steps
            )
            self.num_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
            self.args.num_train_epochs = math.ceil(self.num_train_steps / self.num_update_steps_per_epoch)

            if isinstance(self.args.num_warmup_steps, float):
                self.args.num_warmup_steps = math.ceil(self.args.num_warmup_steps * self.num_train_steps)

            self.total_batch_size = (
                self.args.per_device_train_batch_size
                * self._accelerator.num_processes
                * self.args.gradient_accumulation_steps
            )
            # set learning rate scheduler
            self.lr_scheduler = self._set_scheduler(self.args.num_warmup_steps, self.num_train_steps)

    def _set_scheduler(self, num_warmup_steps, num_train_steps):
        """
        Call after `_accelerator.prepare`
        """
        # define linear/cosine scheduler
        if self.args.scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps * self.args.gradient_accumulation_steps,
                num_training_steps=num_train_steps * self.args.gradient_accumulation_steps,
            )

        elif self.args.scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps * self.args.gradient_accumulation_steps,
                num_training_steps=num_train_steps * self.args.gradient_accumulation_steps,
            )
        return lr_scheduler

    def _init_global_progress_bar(self):
        self.global_prog_bar = tqdm(
            range(self.num_train_steps),
            disable=not self._accelerator.is_main_process,
            position=0,
            leave=True,
        )

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch and return the average loss
        Note: the model must return the loss from its forward function
        """
        self._trn_loss_meter.reset()
        self.model.train()

        for step, batch in enumerate(dataloader):
            with self._accelerator.accumulate(self.model):
                logits, loss = self.model(**batch)
                self._accelerator.backward(loss)
                # assuming dataset has label as key
                self._trn_loss_meter.update(loss.item(), batch["label"].size(0))
                self._accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self._accelerator.sync_gradients:
                    self.global_prog_bar.set_postfix(loss=self._trn_loss_meter.avg)
                    self.global_prog_bar.update(1)

        # can also calculate metrics here for training epoch via `compute_metrics` Callable function
        # ...
        return self._trn_loss_meter.avg

    @torch.no_grad()
    def evaluate(self, dataloader):
        all_logits = []
        all_labels = []
        self._val_loss_meter.reset()
        val_pbar = tqdm(
            range(len(dataloader)),
            disable=not self._accelerator.is_main_process,
            leave=False,
            desc="Running Validation",
        )
        self.model.eval()
        for step, batch in enumerate(dataloader):
            logits, loss = self.model(**batch)
            self._val_loss_meter.update(loss.item(), batch["label"].size(0))
            all_logits.append(self._accelerator.gather_for_metrics(logits).cpu().numpy())
            all_labels.append(self._accelerator.gather_for_metrics(batch["label"]).cpu().numpy())
            val_pbar.update(1)
        val_pbar.close()
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        val_metrics = self.compute_metrics(all_logits, all_labels)

        return EvalOutputs(all_logits, all_labels, val_metrics)

    def save_model(self, path: str, weights_only: Optional[bool] = False, **kwargs):
        self._accelerator.wait_for_everyone()
        model_state_dict = self._accelerator.unwrap_model(self.model).state_dict()
        if weights_only:
            if self._accelerator.is_main_process:
                self._accelerator.save(model_state_dict, path)
            return
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = self.optimizer.state_dict()
        model_dict["scheduler"] = self.lr_scheduler.state_dict()
        model_dict["args"] = self.args

        if self._accelerator.is_main_process:
            self._accelerator.save(
                model_dict,
                path,
            )

    @torch.no_grad()
    def predict(self, checkpoint_path: str):
        self._set_num_workers()
        if self.test_dataset is None or self.test_dataloader is None:
            raise Exception("Either test dataset or test dataloader not found")
        if self.test_dataloader is None:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.args.per_device_val_batch_size,
                shuffle=self.args.test_shuffle,
                collate_fn=self.test_collate_fn,
                sampler=self.test_sampler,
                num_workers=self.args.num_workers,
                drop_last=self.args.test_drop_last,
            )

        model_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(model_dict)
        self._init_accelerator()
        self.model, self.test_dataloader = self._accelerator.prepare(self.model, self.test_dataloader)
        self.model.eval()

        predict_pbar = tqdm(
            range(len(self.test_dataloader)), disable=not self._accelerator.is_main_process, desc="Running Prediction"
        )
        all_logits = []
        for step, batch in self.test_dataloader:
            logits, _ = self.model(**batch)
            all_logits.append(self._accelerator.gather_for_metrics(logits).cpu().numpy())
            predict_pbar.update(1)

        all_logits = np.concatenate(all_logits)
        return all_logits

    def fit(self):
        self._init_trainer()
        self._train_startup_log_msg()
        self._init_global_progress_bar()
        best_loss = -np.inf
        self.best_val_metric = 0
        for epoch in range(self.args.num_train_epochs):
            trn_epoch_loss = self.train_one_epoch(self.train_dataloader)
            eval_outs = self.evaluate(self.val_dataloader)
            self._current_epoch += 1
            self._current_val_metrics.update(eval_outs.metrics)
            self._log_epoch_summary()

            # save best epoch model
            if self.args.save_best_checkpoint:
                metric_value_for_best_model = self._current_val_metrics[self.args.metric_for_best_model]
                if metric_value_for_best_model > self.best_val_metric:
                    self.best_val_metric = metric_value_for_best_model
                    self.save_model(
                        path=os.path.join(self.args.output_dir, "best_model.pth"),
                        weights_only=self.args.save_weights_only,
                    )

        self.global_prog_bar.close()

        # save last model
        if self.args.save_last_checkpoint:
            self.save_model(
                path=os.path.join(self.args.output_dir, "last_model.pth"),
                weights_only=self.args.save_weights_only,
            )

    def _train_startup_log_msg(self):
        self._accelerator.print("***** Running training *****")
        self._accelerator.print(f"  Num examples = {len(self.train_dataset)}")
        self._accelerator.print(f"  Num Epochs = {self.args.num_train_epochs}")
        self._accelerator.print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self._accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}"
        )
        self._accelerator.print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self._accelerator.print(f"  Num warmup steps = {self.args.num_warmup_steps}")
        self._accelerator.print(f"  Total optimization steps = {self.num_train_steps}")

    def _log_epoch_summary(self):
        metrics_summary = []
        for m, v in self._current_val_metrics.items():
            tmp_str = f"val_{m}: {v:.4f}"
            metrics_summary.append(tmp_str)
        metrics_summary = "  |  ".join(metrics_summary)
        summary_str = f"  Epoch {self._current_epoch}  |  train_loss: {self._current_epoch_train_loss:.4f}  |  val_loss: {self._current_epoch_val_loss:.4f}  |  {metrics_summary}"
        if self._accelerator.is_main_process:
            self.global_prog_bar.write(summary_str)
