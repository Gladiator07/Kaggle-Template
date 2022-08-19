import sys
from absl import flags
from ml_collections import config_flags
from accelerate import Accelerator
from utils import set_seed

sys.path.append("../configs")

config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_integer("fold", default=0, help="fold index")
flags.DEFINE_bool("debug", default=False, help="debug pipeline with logging and tracking disabled")
flags.DEFINE_bool("wandb_enabled", default=True, help="enable Weights & Biases logging")

FLAGS = flags.FLAGS
FLAGS(sys.argv)  # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx
cfg = FLAGS.config
debug = FLAGS.debug
fold = FLAGS.fold
wandb_enabled = FLAGS.wandb_enabled


def main():
    # train script here
    set_seed(cfg.seed, cfg.deterministic, cfg.benchmark)

    # init 🤗 Accerelator
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
        mixed_precision=cfg.trainer_args.mixed_precision,
        gradient_accumulation_steps=cfg.trainer_args.gradient_accumulation_steps,
        log_with="wandb" if wandb_enabled else None,
    )

    if wandb_enabled:
        accelerator.init_trackers(project_name=cfg.project, config=cfg.to_dict(), init_kwargs=cfg.wandb_args)

    # rest of the script ....

    # end tracker
    if wandb_enabled:
        accelerator.end_training()


if __name__ == "__main__":
    main()
