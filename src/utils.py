import os
import random
from types import SimpleNamespace

import numpy as np
import slack
import torch


def set_seed(seed: int = 42, deterministic: bool = False, benchmark: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    print(f"Global seed set to {seed}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class SlackNotifier:
    """
    Notifies about important events to a dedicated slack channel
    """

    def __init__(self, slack_token: str, channel: str):
        self.slack_token = slack_token
        self.channel = channel
        self.client = slack.WebClient(token=self.slack_token)

    def notify(self, message: str):
        self.client.chat_postMessage(channel=self.channel, text=message)


def asHours(seconds: float) -> str:
    """
    Returns seconds to human-readable formatted string
    Args:
        seconds (float): total seconds
    Returns:
        str: total seconds converted to human-readable formatted string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"
