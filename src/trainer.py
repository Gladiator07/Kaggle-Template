# Inspired from HuggingFace Trainer and Tez
# https://huggingface.co/docs/transformers/main_classes/trainer
# https://github.com/abhishekkrthakur/tez

import gc
import math
import time
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from utils import AverageMeter, asHours

"""
TODO:
[x] Print valuation metrics nicely after each epoch
[x] Add clip_grad_norm to training loop
[x] Make sure that loss calculated is correct
[x] Maybe pass even the dataloaders to the class to keep the class simple and less cluttered
[x] Add saving and loading of model checkpoint (handle with care in distributed settings)
[x] Maybe wrap up the math around scheduler in a private method
[x] Add a predict method which can perform prediction from loading a model state. Example: `trainer.predict(path, dataloader)`
[x] Check optimization stuff (total training steps, accumulation , lr scheduler)
[ ] Add Weights & Biases logging (handle with care in distributed settings
[ ] Add option to choose between print to console or log to a file (this is not necessary I guess as wandb records all of the stdout)
[ ] Add a `final_summary` method which will print/log the complete summary (best and last metric/loss scores, total time taken, etc, etc)
[ ] Add log_every_n_steps functionality (maybe to reduce bottlenecks)
[ ] Test the code thoroughly with multi-GPU setup
[ ] Test the code thoroughly with TPU setup
[ ] Check the code completely once (if any mistakes, correct them)
[ ] Add type hints
[ ] Add docstrings 😛
"""


@dataclass
class EvalOutput:
    """
    A dataclass to store outputs from `Trainer.evaluate` method
    Change this class and the method if you want to return something else
    """

    logits: np.ndarray
    labels: np.ndarray
    metrics: Dict[str, float]
    loss: float


@dataclass
class TrainerArguments:
    output_dir: str

    num_train_epochs: Optional[int] = 5
    gradient_accumulation_steps: Optional[int] = 1
    max_grad_norm: Optional[float] = 1.0
    mixed_precision: Optional[str] = None  # bf16, fp16

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
        # passing from outside for now to have the flexibility to modify param groups
        optimizer: torch.optim = None,
        # 🤗 `accelerator` can be passed to the class if 🤗 `accelerator` if initialized at the start of the train script.
        # If not passed will be initialzed internally by the `Trainer`.
        # Recommended way is to initialize the accelerator at the start of the main train script
        # and pass as an argument to this class
        accelerator: Optional[Accelerator] = None,
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

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.compute_metrics = compute_metrics

        # internals
        self._trn_loss_meter = AverageMeter("train_loss", ":.4e")
        self._val_loss_meter = AverageMeter("val_loss", ":.4e")
        self._current_epoch = 0
        self._epoch_time = 0
        self._current_epoch_train_loss = 0.0
        self._current_epoch_val_loss = 0.0
        self._current_val_metrics = {}
        self._best_val_metric = 0
        self._accelerator = accelerator

    def _init_accelerator(self, *args, **kwargs):
        """
        Initialize the 🤗 Accelerator
        """
        self._accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=self.args.mixed_precision if self.args.mixed_precision is not None else "no",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            *args,
            **kwargs,
        )

    def _init_trainer(self):
        """
        Initialize trainer for training and evaluation
        """
        # create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        # extract batch size and length of dataset from dataloader before sharding (for logging purposes)
        self.per_device_train_batch_size = self.train_dataloader.batch_size
        self.total_samples = len(self.train_dataloader.dataset)

        self._init_accelerator()

        # not putting any checks if train_dataloader or val_dataloader is present or not
        # as this method is explicitly used by `.fit()` which requires both dataloaders

        # prepare for distributed training aka put everything in 🤗 Accelerate 🚀
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        ) = self._accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.val_dataloader)

        # TODO: handle weight tying of model after pushed to XLA device here
        # https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks

        # re-calculate total training steps and epoch as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        self.num_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.num_train_steps / self.num_update_steps_per_epoch)

        if isinstance(self.args.num_warmup_steps, float):
            self.args.num_warmup_steps = math.ceil(self.args.num_warmup_steps * self.num_train_steps)

        self.total_batch_size = (
            self.per_device_train_batch_size * self._accelerator.num_processes * self.args.gradient_accumulation_steps
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

    def train_one_epoch(self, dataloader: DataLoader):
        """
        Trains the model for one epoch and return the average loss
        Note: the model must return the loss from its forward function
        """

        self._trn_loss_meter.reset()
        self.model.train()

        for step, batch in enumerate(dataloader):
            with self._accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                _, loss = self.model(**batch)
                self._accelerator.backward(loss)
                if self._accelerator.sync_gradients:
                    self._accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                # assuming dataset has label as key
                step_loss_gathered = self._accelerator.gather(loss).mean().item()
                self._trn_loss_meter.update(
                    step_loss_gathered * self.args.gradient_accumulation_steps, batch["label"].size(0)
                )
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
            step_loss_gathered = self._accelerator.gather(loss).mean().item()
            self._val_loss_meter.update(step_loss_gathered, batch["label"].size(0))
            all_logits.append(self._accelerator.gather_for_metrics(logits).cpu().numpy())
            all_labels.append(self._accelerator.gather_for_metrics(batch["label"]).cpu().numpy())
            val_pbar.update(1)
        val_pbar.close()
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        val_metrics = self.compute_metrics(all_logits, all_labels)

        return EvalOutput(all_logits, all_labels, val_metrics, self._val_loss_meter.avg)

    def fit(self):
        """
        Main entry point of training and evaluation routines
        """
        self._init_trainer()
        self._train_startup_log_msg()
        self._init_global_progress_bar()
        self.best_val_metric = 0
        for epoch in range(self.args.num_train_epochs):
            epoch_start_time = time.time()
            trn_epoch_loss = self.train_one_epoch(self.train_dataloader)
            eval_outs = self.evaluate(self.val_dataloader)
            self._current_epoch += 1
            self._current_val_metrics.update(eval_outs.metrics)
            self._current_epoch_train_loss = trn_epoch_loss
            self._current_epoch_val_loss = eval_outs.loss

            # save best model
            if self.args.save_best_checkpoint:
                metric_value_for_best_model = self._current_val_metrics[self.args.metric_for_best_model]
                if metric_value_for_best_model > self.best_val_metric:
                    self.best_val_metric = metric_value_for_best_model
                    self.save_model(
                        path=os.path.join(self.args.output_dir, "best_model.bin"),
                        weights_only=self.args.save_weights_only,
                    )
            self._gentle_cleanup()

            self._epoch_time = asHours(time.time() - epoch_start_time)
            self._log_epoch_summary()

        self.global_prog_bar.close()
        # save last model
        if self.args.save_last_checkpoint:
            self.save_model(
                path=os.path.join(self.args.output_dir, "last_model.bin"),
                weights_only=self.args.save_weights_only,
            )

        self._full_cleanup()

    @torch.no_grad()
    def predict(self, checkpoint_path: str, test_dataloader: Optional[DataLoader] = None):
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        if self.test_dataloader is None:
            raise Exception(
                "`.predict()` method requires a dataloader, either pass it through the `.predict()` method or while initiating the `Trainer` class"
            )

        if self._accelerator is not None:
            # unwrap the model if distributed training `.fit()` was performed before calling `.predict()`
            self.model = self._accelerator.unwrap_model(self.model)
            self._accelerator = None

        # reinit accelerator
        self._init_accelerator()

        # load model
        if self._accelerator.is_main_process:
            model_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(model_dict)

        # init and prepare accelerator
        self.model, self.test_dataloader = self._accelerator.prepare(self.model, self.test_dataloader)
        self.model.eval()

        predict_pbar = tqdm(
            range(len(self.test_dataloader)), disable=not self._accelerator.is_main_process, desc="Running Prediction"
        )
        all_logits = []
        for step, batch in enumerate(self.test_dataloader):
            logits, _ = self.model(**batch)
            all_logits.append(self._accelerator.gather_for_metrics(logits).cpu().numpy())
            predict_pbar.update(1)
        predict_pbar.close()
        all_logits = np.concatenate(all_logits)
        return all_logits

    def save_model(self, path: str, weights_only: Optional[bool] = False):
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

    def _train_startup_log_msg(self):
        self._accelerator.print("***** Running training *****")
        self._accelerator.print(f"  Num Samples = {self.total_samples}")
        self._accelerator.print(f"  Num Epochs = {self.args.num_train_epochs}")
        self._accelerator.print(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        self._accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}"
        )
        self._accelerator.print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self._accelerator.print(f"  Num warmup steps = {self.args.num_warmup_steps}")
        self._accelerator.print(f"  Total optimization steps = {self.num_train_steps}")

    def _log_epoch_summary(self):
        sepr = " | "
        metrics_summary = []
        for m, v in self._current_val_metrics.items():
            tmp_str = f"val_{m}: {v:.4f}"
            metrics_summary.append(tmp_str)
        metrics_summary = f"{sepr}".join(metrics_summary)
        summary_str = (
            f"  Epoch {self._current_epoch}"
            + sepr
            + f"train_loss: {self._current_epoch_train_loss:.4f}"
            + sepr
            + f"val_loss: {self._current_epoch_val_loss:.4f}"
            + sepr
            + metrics_summary
            + sepr
            + f"time: {self._epoch_time}"
        )
        if self._accelerator.is_main_process:
            self.global_prog_bar.write(summary_str)

    def _gentle_cleanup(self):
        gc.collect()

    def _full_cleanup(self):
        self._accelerator.clear()
