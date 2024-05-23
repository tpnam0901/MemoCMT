import importlib
import sys

from configs.base import BaseConfig


def get_options(
    path: str,
) -> BaseConfig:
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    options = config.Config()
    return options
