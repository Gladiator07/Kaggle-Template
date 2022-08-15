from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainerArguments:
    train_batch_size: Optional[int] = 32
    valid_batch_size: Optional[int] = 32

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
    step_scheduler_after: Optional[str] = "epoch"  # step

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
[] Create dataloaders internally by `TrainerArguments` and passed dataset classes (add collate_fn, shuffle, etc, basically all parameters accepted by Dataloader)
[] Whether dataloaders should be created internally or should be provided as args ??? [Decide on this]
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
        pass

    def _init_accelerator(self, *args, **kwargs):
        pass

    def _init_trainer(self, *args, **kwargs):
        pass

    def train_one_step(self, **kwargs):
        pass

    def train_one_epoch(self, **kwargs):
        pass

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def save_model(self, weights_only: Optional[False], **kwargs):
        # make sure to handle distributed case here
        pass

    def load_model(self, weights_only: Optional[False], **kwargs):
        # make sure to handle distributed case here
        pass

    def predict(self, **kwargs):
        # make sure to handle distributed case here
        pass
