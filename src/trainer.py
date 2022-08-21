# Inspired from HuggingFace Trainer and Tez
# https://huggingface.co/docs/transformers/main_classes/trainer
# https://github.com/abhishekkrthakur/tez

from typing import Callable, Dict, Optional, Union
import os
import gc
import math
import time
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from utils import AverageMeter, asHours


@dataclass
class EvalOutput:
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

    num_train_epochs: Optional[int] = 5
    gradient_accumulation_steps: Optional[int] = 1
    max_grad_norm: Optional[float] = 1.0
    mixed_precision: Optional[str] = None  # bf16, fp16

    # scheduler args
    # TODO: add support for a custom scheduler and stepping after epoch or batch option
    scheduler_type: Optional[str] = "cosine"  # linear
    num_warmup_steps: Optional[Union[float, int]] = 0.1

    # misc
    log_every_n_steps: Optional[int] = 10
    save_best_checkpoint: Optional[bool] = True
    save_last_checkpoint: Optional[bool] = True
    save_weights_only: Optional[bool] = True
    metric_for_best_model: Optional[str] = "accuracy"
    load_best_model_at_end: Optional[bool] = False


"""
A boiler plate Trainer class which can be used with minimal to no code changes for most of the problems
For now no plan to make it fully generalizable, just copy paste this file and modify according to the project/competition if required
"""


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,  # model is required
        args: Optional[TrainerArguments] = None,
        # passing from outside for now to have the flexibility to modify param groups
        optimizer: Optional[torch.optim.Optimizer] = None,
        accelerator: Optional[Accelerator] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        NOTE:
        🤗 `accelerator` should be initialized at the start of the `main()` function or train script
        and then passed to the initialization of this class. This design decision is made because
        dataset preparation/logging will be done multiple times on all cores in distributed settings (especially on TPUs).
        To have the flexibility of doing it on the main process and leveraging the functionalities of 🤗 Accelerate,
        accelerator should be initialized at the start of the training script.
        Also, logging to W&B is more convenient as it can be initialized at the start of the script to capture all console logs
        and not have to wait until `trainer.fit()`
        """

        self.model = model
        self.args = args
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.compute_metrics = compute_metrics

        # internals
        self.best_val_metric = 0
        self.accelerator = accelerator

        self._trn_loss_meter = AverageMeter("train_loss", ":.4e")
        self._val_loss_meter = AverageMeter("val_loss", ":.4e")
        self._current_epoch = 0
        self._global_step = 0
        self._epoch_time = 0
        self._start_time = 0
        self._current_epoch_train_loss = 0.0
        self._current_epoch_val_loss = 0.0
        self._current_val_metrics = {}
        if self.accelerator is not None:
            self._wandb = True if self.accelerator.log_with != [] else False

    def _init_accelerator(self, *args, **kwargs):
        """
        Initializes the 🤗 Accelerator
        Note: Only use this in `.predict()` method, not intended to be used in `.fit()`
        For training/evaluation accelerator should be passed from outside
        Might remove this for `.predict() method too in the future
        """
        self.accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=self.args.mixed_precision if self.args.mixed_precision is not None else "no",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            *args,
            **kwargs,
        )

    def _init_trainer(self):
        """
        Initializes trainer for training and evaluation
        """
        # create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        # extract batch size and length of dataset from dataloader before sharding (for logging purposes)
        self.per_device_train_batch_size = self.train_dataloader.batch_size
        self.total_samples = len(self.train_dataloader.dataset)

        if self.accelerator is None:
            raise Exception(
                "🤗 Accelerator object not found, it is required while calling `.fit()` method, pass it to the class' init"
            )

        # not putting any checks if train_dataloader or val_dataloader is present or not
        # as this method is explicitly used by `.fit()` which requires both dataloaders
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        self.num_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        if isinstance(self.args.num_warmup_steps, float):
            self.args.num_warmup_steps = math.ceil(self.args.num_warmup_steps * self.num_train_steps)
        self.lr_scheduler = self._set_scheduler(
            num_warmup_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps,
            num_train_steps=self.num_train_steps * self.args.gradient_accumulation_steps,
        )

        # prepare for distributed training aka put everything in 🤗 Accelerate 🚀
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )

        # TODO: handle weight tying of model after pushed to XLA device here
        # https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks

        # re-calculate total training steps as length of dataloader might have changed
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        self.num_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.num_train_steps / num_update_steps_per_epoch)

        self.total_batch_size = (
            self.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )

    def _set_scheduler(self, num_warmup_steps: int, num_train_steps: int):
        """
        Call after `accelerator.prepare` and calculating the total train steps after `prepare`
        """

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
        """
        Global Progress bar to show the progress of the complete training
        """
        self.global_prog_bar = tqdm(
            range(self.num_train_steps),
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

    def train_one_epoch(self, dataloader: DataLoader):
        """
        Trains the model for one epoch and returns the average loss
        Note: the model must return the loss from its forward function
        """

        self._trn_loss_meter.reset()
        self.model.train()

        for batch in dataloader:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)
                _, loss = self.model(**batch)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                # gather and average loss across all processes
                step_loss_gathered = self.accelerator.gather(loss).mean().item()
                self._trn_loss_meter.update(
                    step_loss_gathered * self.args.gradient_accumulation_steps, batch["labels"].size(0)
                )
            if self.accelerator.sync_gradients:
                self.global_prog_bar.set_postfix(loss=self._trn_loss_meter.avg)
                self.global_prog_bar.update(1)
                self._global_step += 1

                if self._wandb:
                    self.accelerator.log({"train/loss": self._trn_loss_meter.val}, step=self._global_step)

        # log average epoch loss
        if self._wandb:
            self.accelerator.log({"train/loss_epoch": self._trn_loss_meter.avg})

        # can also calculate metrics here for training epoch via `compute_metrics` Callable function
        # ...
        return self._trn_loss_meter.avg

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        all_logits = []
        all_labels = []
        self._val_loss_meter.reset()
        val_pbar = tqdm(
            range(len(dataloader)),
            disable=not self.accelerator.is_main_process,
            leave=False,
            desc="Running Validation",
        )
        self.model.eval()
        for batch in dataloader:
            # required if `.evaluate()` is called independently
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            logits, loss = self.model(**batch)
            step_loss_gathered = self.accelerator.gather(loss).mean().item()
            self._val_loss_meter.update(step_loss_gathered, batch["labels"].size(0))
            all_logits.append(self.accelerator.gather_for_metrics(logits).cpu().numpy())
            all_labels.append(self.accelerator.gather_for_metrics(batch["labels"]).cpu().numpy())
            val_pbar.update(1)
        val_pbar.close()
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        val_metrics = self.compute_metrics(all_logits, all_labels)
        val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}

        if self._wandb:
            self.accelerator.log({"val/loss": self._val_loss_meter.avg})
            self.accelerator.log(val_metrics)
        return EvalOutput(all_logits, all_labels, val_metrics)

    def fit(self):
        """
        Main entry point of training and evaluation routines
        """
        self._start_time = time.time()
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
            self._current_epoch_val_loss = self._val_loss_meter.avg

            # save best model
            if self.args.save_best_checkpoint:
                metric_value_for_best_model = self._current_val_metrics[self.args.metric_for_best_model]
                if metric_value_for_best_model > self.best_val_metric:
                    self.best_val_metric = metric_value_for_best_model
                    self.save_model(
                        path=os.path.join(self.args.output_dir, "best_model.bin"),
                        weights_only=self.args.save_weights_only,
                    )

            self._epoch_time = asHours(time.time() - epoch_start_time)
            self._log_epoch_summary()

        self.global_prog_bar.close()
        # save last model
        if self.args.save_last_checkpoint:
            self.save_model(
                path=os.path.join(self.args.output_dir, "last_model.bin"),
                weights_only=self.args.save_weights_only,
            )

        # if load_best_model is True
        if self.args.load_best_model_at_end:
            self.load_model(weights_only=True, load_best_model=True)

        self._train_end_log_msg()
        self._full_cleanup()

    @torch.no_grad()
    def predict(self, checkpoint_path: Union[str, Path], test_dataloader: Optional[DataLoader] = None):
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        if self.test_dataloader is None:
            raise Exception(
                "`.predict()` method requires a dataloader, either pass it through the `.predict()` method or while initiating the `Trainer` class"
            )

        if self.accelerator is not None:
            # unwrap the model if distributed training `.fit()` was performed before calling `.predict()`
            # TODO: following line can introduce bug if accelerator is passed from outside
            # check this once
            self.model = self.accelerator.unwrap_model(self.model)
            self.accelerator = None

        # turn off wandb for inference (generally this function is used offline on Kaggle)
        self._wandb = False
        # reinit accelerator
        self._init_accelerator()

        # load best or last model
        self.load_model(checkpoint_path, weights_only=True)

        # init and prepare accelerator
        self.model, self.test_dataloader = self.accelerator.prepare(self.model, self.test_dataloader)
        self.model.eval()

        predict_pbar = tqdm(
            range(len(self.test_dataloader)), disable=not self.accelerator.is_main_process, desc="Running Prediction"
        )
        all_logits = []
        for batch in self.test_dataloader:
            logits, _ = self.model(**batch)
            all_logits.append(self.accelerator.gather_for_metrics(logits).cpu().numpy())
            predict_pbar.update(1)
        predict_pbar.close()
        all_logits = np.concatenate(all_logits)
        return all_logits

    def save_model(self, path: Union[str, Path], weights_only: Optional[bool] = False):
        self.accelerator.wait_for_everyone()
        model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        if weights_only:
            if self.accelerator.is_main_process:
                self.accelerator.save(model_state_dict, path)
            return
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = self.optimizer.state_dict()
        model_dict["scheduler"] = self.lr_scheduler.state_dict()
        model_dict["args"] = self.args

        if self.accelerator.is_main_process:
            self.accelerator.save(
                model_dict,
                path,
            )

    def load_model(
        self,
        path: Union[str, Path] = None,
        weights_only: Optional[bool] = False,
        load_best_model: Optional[bool] = False,
    ):
        """
        Loads the model from the given path (pass best_model as True to load best checkpoint)
        """
        if path is None:
            if load_best_model:
                path = os.path.join(self.args.output_dir, "best_model.bin")
            else:
                # load last model
                path = os.path.join(self.args.output_dir, "last_model.bin")

        if self.accelerator.is_main_process:
            model_dict = torch.load(path, map_location="cpu")
            if weights_only:
                self.accelerator.unwrap_model(self.model).load_state_dict(model_dict)
            else:
                self.accelerator.unwrap_model(self.model).load_state_dict(model_dict["state_dict"])
                self.optimizer.load_state_dict(model_dict["optimizer"])

    def _train_startup_log_msg(self):
        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Mixed Precision = {self.args.mixed_precision}")
        self.accelerator.print(f"  Num Samples = {self.total_samples}")
        self.accelerator.print(f"  Num Epochs = {self.args.num_train_epochs}")
        self.accelerator.print(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        self.accelerator.print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}"
        )
        self.accelerator.print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.accelerator.print(f"  Num warmup steps = {self.args.num_warmup_steps}")
        self.accelerator.print(f"  Total optimization steps = {self.num_train_steps}")

    def _train_end_log_msg(self):
        total_elapsed_time = time.time() - self._start_time
        self.accelerator.print(f"\n\n===== Training completed in {asHours(total_elapsed_time)} =====\n\n")

    def _log_epoch_summary(self):
        sepr = " | "
        metrics_summary = []
        for m, v in self._current_val_metrics.items():
            tmp_str = f"{m}: {v:.4f}"
            metrics_summary.append(tmp_str)
        metrics_summary = f"{sepr}".join(metrics_summary)
        summary_str = (
            f"  Epoch {self._current_epoch}"
            + sepr
            + f"train/loss: {self._current_epoch_train_loss:.4f}"
            + sepr
            + f"val/loss: {self._current_epoch_val_loss:.4f}"
            + sepr
            + metrics_summary
            + sepr
            + f"time: {self._epoch_time}"
        )
        if self.accelerator.is_main_process:
            self.global_prog_bar.write(summary_str)

    def _cleanup(self):
        gc.collect()

    def _full_cleanup(self):
        self.accelerator.clear()
        gc.collect()
        torch.cuda.empty_cache()
