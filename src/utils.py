import os
import random
from types import SimpleNamespace

import numpy as np
import slack
import torch


def set_seed(
    seed: int = 42, deterministic: bool = False, benchmark: bool = False
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    print(f"Global seed set to {seed}")


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
