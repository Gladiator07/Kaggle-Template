import os
from default_config import get_default_config

cfg = get_default_config()
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.seed = 7

# data
cfg.datamodule = dict()

# model
cfg.model = dict()

# trainer
cfg.trainer = dict(
    accelerator="gpu",
    accumulate_grad_batches=1,
    auto_lr_find=False,
    benchmark=False,
    deterministic=False,  # "warn"
    fast_dev_run=False,
    gradient_clip_val=1.0,
    overfit_batches=0.0,
    precision=16,
    max_epochs=5,
    val_check_interval=1.0,  # float in range [0.0, 1.0] to check after fraction of epoch
)

# tracking
# override version number from cmd for every new experiment for this config
cfg.version = 1
cfg.experiment_name = f"{cfg.version}_{cfg.name}"
# override from cmd for short overview of the experiment
cfg.notes = "baseline experiment"


def get_config():
    return cfg
