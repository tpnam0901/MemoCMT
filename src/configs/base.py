import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union
import importlib
import sys


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, save_folder: str):
        pass

    @abstractmethod
    def load(self, cfg_path: str):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, save_folder: str):
        message = "\n"
        for k, v in sorted(vars(self).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(save_folder), exist_ok=True)
        out_cfg = os.path.join(save_folder, "cfg.log")
        with open(out_cfg, "w") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            value_converted = None
            if "." in value and value.replace(".", "").isdigit():
                value_converted = float(value)
            elif value.isdigit():
                value_converted = int(value)
            elif value == "True":
                value_converted = True
            elif value_converted == "False":
                value_converted = False
            elif (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                value_converted = value[1:-1]
            else:
                value_converted = value
            return value_converted

        with open(cfg_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = (
                    data[i].split(":")[0].strip(),
                    data[i].split(":")[1].strip(),
                )
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [decode_value(x) for x in value]
                else:
                    value = decode_value(value)

                data_dict[key] = value
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # Training settings
        self.epochs: int = 250
        self.batch_size: int = 32
        self.checkpoint_dir: "str" = "working/checkpoints"
        self.ckpt_save_fred: int = 4000

        # Optim settings
        self.learning_rate: float = 0.0001
        self.weight_decay: float = 0.0001
        self.lr_step_size: int = 50
        self.gamma: float = 0.1

        # Resume training
        self.resume: bool = False  # map to "resume"
        # path to checkpoint.pt file, only available when using save_all_states = True in previous training
        self.resume_path: str = ""
        self.opt_path: str = ""

        # Model settings
        self.model_type: str = "BirdClassification"  # [BirdClassification, PANNS]
        self.model_infer: bool = False
        self.audio_encoder_type: str = "hubert_base"  # hubert_base, wavlm_base
        self.audio_dim: int = 768
        self.linears: List = [256]
        self.num_classes: int = -1

        # PANNS model settings
        self.panns_type: str = "Wavegram_Logmel_Cnn14"
        self.window_size: int = 1024
        self.hop_size: int = 320
        self.mel_bins: int = 64
        self.fmin: int = 125  # 50
        self.fmax: int = 7500  # 14000

        # Dataset
        self.data_root: str = "working/birdclef-2024/train_audio"
        self.data_type: str = "waveform"  # waveform, log_mel
        self.num_workers: int = 8  # map to "workers"
        self.max_audio_sec: int = 5
        self.sample_rate: int = 16000

        for key, value in kwargs.items():
            setattr(self, key, value)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
