import os
from default_config import get_default_config

cfg = get_default_config()
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.seed = 7
cfg.deterministic = False
cfg.benchmark = False
# data


# model
cfg.model = dict()

# trainer
cfg.trainer_args = dict(
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    mixed_precision="fp16",  # bf16, no
    scheduler_type="cosine",  # linear
    num_warmup_steps=0.1,  # exact steps (int) can be also passed
    save_best_checkpoint=True,
    save_last_checkpoint=True,
    save_weights_only=True,
    metric_for_best_model="val/accuracy",
)

# tracking
# override version number from cmd for every new run for this config
cfg.version = 0
cfg.experiment_name = f"{cfg.name}_v{cfg.version}"
# override from cmd for short overview of the experiment
cfg.notes = "baseline experiment"

cfg.wandb_args = dict(group=cfg.experiment_name, notes=cfg.notes, save_code=True)


def get_config():
    return cfg
