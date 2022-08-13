import sys
from absl import app, flags
from ml_collections import config_flags

sys.path.append("../configs")

config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_boolean(
    "debug", default=False, help="debug pipeline with logging and tracking disabled"
)

FLAGS = flags.FLAGS


def main(argv):
    del argv
    cfg = FLAGS.config
    debug = FLAGS.debug


if __name__ == "__main__":
    app.run(main)
