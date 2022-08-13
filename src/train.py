import sys
from absl import app, flags
from ml_collections import config_flags

sys.path.append("../configs")

config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_integer("fold", default=0, help="fold index")
flags.DEFINE_bool(
    "debug", default=False, help="debug pipeline with logging and tracking disabled"
)
flags.DEFINE_bool("wandb_enabled", default=True, help="enable Weights & Biases logging")

FLAGS = flags.FLAGS
cfg = FLAGS.config
debug = FLAGS.debug
fold = FLAGS.fold
wandb_enabled = FLAGS.wandb_enabled


def main(argv):
    del argv
    # train script here


if __name__ == "__main__":
    app.run(main)
