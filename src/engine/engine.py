# Inspired and modified from: https://github.com/abhishekkrthakur/tez

from dataclasses import dataclass
import multiprocessing
from typing import Optional, Union
from xml.sax import parseString

import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(val={self.val}, avg={self.avg}, sum={self.sum}, count={self.count})"


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
------
DESIGN
------
[x] Create dataloaders internally by `TrainerArguments` and passed dataset classes (add collate_fn, shuffle, etc, basically all parameters accepted by Dataloader)
[x] Whether dataloaders should be created internally or should be provided as args ??? [Decide on this]
[] `log_mode`: choose whether to log everything to a log file or just print to stdout
[] Implement log_every_n_steps (maybe average over n steps or just log the value, need to decide on this)
[] Implement saving of best and last checkpoint functionality with appropiate file names
[] Refer TPU guide to make this trainer work out-of-the-box with TPUs (https://huggingface.co/docs/accelerate/concept_guides/training_tpu)
[] Global Progress bar showing total steps and will display loss in postfix (No progress bar per epoch, however a sub progress bar for evaluation phase) [Very similar design as 🤗 Trainer]
[] Print/Log metrics after each evaluation phase
[] If possible pass a `compute_metrics` function to compute the metrics after each evaluation phase (similar to 🤗 Trainer)
[] Evaluation strategy: epoch or steps (first implement for epoch, steps maybe later)
[] Also implement `num_train_steps` which will train for the given number of steps, if this argument is provided then it will overwrite the `num_train_epochs`
[] Find an efficient way to bind optimizer and scheduler to the trainer (maybe define the name and arguments from TrainerArguments and use getattr for initiating)
[] Add support for custom scheduler (now the trainer only supports linear and cosine decay with warmup)
[] Add enums and internal state to the class as in Tez
[] Add a central metrics monitoring system which will monitor the metrics returned by `compute_metrics` function
-------
GOTCHAS
-------
[] Scheduler needs to be instantiated inside the Trainer class because dataloader might change lengths in distributed settings[- Optimizer is passed from outside for now, maybe integrated in the class if possible
[] Make sure to calculate total training steps for scheduler after `accelerator.prepare`
[] Handle tying and un-tying of weights in case of TPU
[] Implement W&B tracking carefully in distributed settings
"""


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,  # passing from outside for now to have the flexibility to modify param groups
        args: Optional[TrainerArguments],
    ):

        self.model = model
        self.optimizer = optimizer
        self.args = args

    def _init_accelerator(self, *args, **kwargs):
        self.accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=self.args.mixed_precision
            if self.args.mixed_precision is not None
            else "no",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def _init_trainer(
        self,
        train_dataset: Optional[Dataset],
        valid_dataset: Optional[Dataset],
        **kwargs,
    ):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_sampler = kwargs.get("train_sampler", None)
        self.valid_sampler = kwargs.get("valid_sampler", None)
        self.train_collate_fn = kwargs.get("train_collate_fn", None)
        self.valid_collate_fn = kwargs.get("valid_collate_fn", None)

        if self.args.num_workers == -1:
            self.args.num_workers = multiprocessing.cpu_count()
            if self.args.num_workers > 4:
                self.args.num_workers -= 2

        # if not provided in the fit/validate function, then create dataloaders
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

        if self.valid_dataloader is None:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
            ) = self.accelerator.prepare(
                self.model, self.train_dataloader, self.optimizer
            )

        else:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.valid_dataloader
            )

        # now calculate total training steps as prepare method may change its length
        if isinstance(self.args.num_warmup_steps, float):
            num_warmup_steps = 0.1 * len(self.train_dataloader)
        elif isinstance(self.args.num_warmup_steps, int):
            num_warmup_steps = self.args.num_warmup_steps

        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs

        if self.args.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        elif self.args.scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.progress_bar = tqdm(
            range(self.args.num_train_epochs * len(self.train_dataloader)),
            disable=not self.accelerator.is_main_process,
        )

    def set_train_epoch_start(self, dataloader):
        try:
            self.train_loader_bs = dataloader.batch_sampler.batch_size
        except AttributeError:
            self.train_loader_bs = dataloader._loader.batch_sampler.batch_size
        self.model.train()

    def train_one_step(self, data):
        with self.accelerator.accumulate(self.model):
            _, loss = self.model(**data)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)
        return loss

    def train_one_epoch(self, dataloader):
        # set start of training epoch (maybe when enums are implemented)
        self.set_training_epoch_start(dataloader)
        losses = AverageMeter()
        for batch_idx, data in enumerate(dataloader):
            loss = self.train_one_step(data)
            # TODO: check difference between gather and this stuff
            losses = losses.update(
                loss.item() * self.config.gradient_accumulation_steps,
                self.train_loader_bs,
            )

        return losses.avg

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, **kwargs):
        self._init_trainer(train_dataset, valid_dataset)

        for _ in range(self.args.num_train_epochs):
            trn_epoch_loss = self.train_one_epoch(self.train_dataloader)

            # TODO: add a check for val strategy as epoch

    def set_validation_epoch_start(self, dataloader):
        try:
            self.valid_loader_bs = dataloader.batch_sampler.batch_size
        except AttributeError:
            self.valid_loader_bs = dataloader._loader.batch_sampler.batch_size
        self.model.eval()

    def predict_step(self, data):
        _, loss = self.model(data)
        return loss

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.set_validation_epoch_start()
        losses = AverageMeter()
        for batch_index, data in enumerate(dataloader):
            loss = self.predict_step(data)
        losses.update(loss.item(), self.valid_loader_bs)

    def save_model(self, weights_only: Optional[False], **kwargs):
        # make sure to handle distributed case here
        pass

    def load_model(self, weights_only: Optional[False], **kwargs):
        # make sure to handle distributed case here
        pass

    def predict(self, **kwargs):
        # make sure to handle distributed case here
        pass
