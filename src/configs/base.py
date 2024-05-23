import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, cfg):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, cfg):
        message = "\n"
        for k, v in sorted(vars(cfg).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(cfg.checkpoint_dir), exist_ok=True)
        out_opt = os.path.join(cfg.checkpoint_dir, "cfg.log")
        with open(out_opt, "w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            convert_value = None
            if "." in value and value.replace(".", "").isdigit():
                convert_value = float(value)
            elif value.isdigit():
                convert_value = int(value)
            elif value == "True":
                convert_value = True
            elif value == "False":
                convert_value = False
            elif value == "None":
                convert_value = None
            elif (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                convert_value = value[1:-1]
            else:
                convert_value = value
            return convert_value

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
        self.trainer = "Trainer"  # Trainer type use for training model [MSER_Trainer, Trainer, MarginTrainer]
        self.num_epochs: int = 100
        self.checkpoint_dir: str = "checkpoints"
        self.save_all_states: bool = False
        self.save_best_val: bool = True
        self.max_to_keep: int = 1
        self.save_freq: int = 100000
        self.batch_size: int = 1

        # Learning rate
        self.learning_rate: float = 0.0001
        self.learning_rate_step_size: int = 30
        self.learning_rate_gamma: float = 0.1

        self.optimizer_type: str = "SGD"  # Adam, SGD, AdamW
        # Adam config
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0
        # SGD config
        self.momemtum = 0.99
        self.sdg_weight_decay = 1e-6

        # Resume training
        self.resume: bool = False
        # path to checkpoint.pt file, only available when using save_all_states = True in previous training
        self.resume_path: Union[str, None] = None
        self.cfg_path: Union[str, None] = None
        if self.resume:
            assert os.path.exists(str(self.resume_path)), "Resume path not found"

        self.loss_type: str = "CrossEntropyLoss"

        # Dataset
        self.data_name: str = (
            "IEMOCAP"  # [IEMOCAP, ESD, MELD, IEMOCAPAudio, IEMOCAP_MSER]
        )
        self.data_root: str = (
            "data/IEMOCAP_preprocessed"  # folder contains train.pkl and test.pkl
        )
        self.data_valid: str = (
            "val.pkl"  # change this to your validation subset name if you want to use validation dataset. If None, test.pkl will be use
        )
        self.num_workers = 0

        # use for training with batch size > 1
        self.text_max_length: int = 297
        self.audio_max_length: int = 546220

        # Model
        self.num_classes: int = 4
        self.num_attention_head: int = 8
        self.dropout: float = 0.5
        self.model_type: str = "TestSER"  #
        self.text_encoder_type: str = "bert"  # [bert, roberta]
        self.text_encoder_dim: int = 768
        self.text_unfreeze: bool = False
        self.audio_encoder_type: str = "focalnet_t"
        self.audio_im_size: int = 224
        self.audio_encoder_dim: int = 768
        self.audio_unfreeze: bool = False

        self.fusion_dim: int = 768
        self.fusion_head_output_type: str = "cls"  # [cls, mean, max]

        # Search for linear layer output dimension
        self.linear_layer_output: List = [128]
        self.linear_layer_last_dim: int = 64
        for key, value in kwargs.items():
            setattr(self, key, value)
