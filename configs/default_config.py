import os
import pkgutil
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT = ""


def determine_env_and_set_root_paths(cfg: SimpleNamespace) -> SimpleNamespace:
    """
    Determines current environment and set root directory paths
    """
    project = cfg.project

    if "google.colab" in sys.modules:
        cfg.environ = "colab"
        cfg.code_dir = Path(f"/content/{project}")
        cfg.artifacts_dir = Path(f"/content/artifacts")

    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""):
        cfg.environ = "kaggle"
        cfg.code_dir = Path(f"/kaggle/working/{project}")
        cfg.artifacts_dir = Path(f"/kaggle/working/artifacts")

    elif pkgutil.find_loader("jarviscloud"):
        cfg.environ = "jarvislabs"
        cfg.code_dir = Path(f"/home/{project}")
        cfg.artifacts_dir = Path(f"/home/artifacts")

    else:
        cfg.environ = "local"

    return cfg


# default config
cfg = SimpleNamespace(**{})
cfg.project = PROJECT
cfg = determine_env_and_set_root_paths(cfg)

basic_config = cfg
