import os
import pkgutil
import sys
from pathlib import Path

import ml_collections as mlc

PROJECT = ""


def determine_env_and_set_root_paths(cfg: mlc.ConfigDict) -> mlc.ConfigDict:
    """
    Determines current environment and set root directory paths
    """
    project = cfg.project

    if "google.colab" in sys.modules:
        cfg.environ = "colab"
        cfg.code_dir = Path(f"/content/{project}")
        cfg.artifacts_dir = Path("/content/artifacts")

    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""):
        cfg.environ = "kaggle"
        cfg.code_dir = Path(f"/kaggle/working/{project}")
        cfg.artifacts_dir = Path("/kaggle/working/artifacts")

    elif pkgutil.find_loader("jarviscloud"):
        cfg.environ = "jarvislabs"
        cfg.code_dir = Path(f"/home/{project}")
        cfg.artifacts_dir = Path("/home/artifacts")

    else:
        # raise NotImplementedError("Current Environment is not recognized")
        cfg.environ = "local"
    return cfg


# default config
cfg = mlc.ConfigDict()
cfg.project = PROJECT
cfg = determine_env_and_set_root_paths(cfg)

# add default config specific to project here
# ...


def get_default_config():
    return cfg
