from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 100

        self.loss_type = "CrossEntropyLoss"

        self.checkpoint_dir = "working/checkpoints/MELD"

        self.model_type = "TestSER"

        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "focalnet_t"
        self.audio_im_size: int = 224
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.fusion_dim: int = 768

        self.num_classes = 7

        # Dataset
        self.data_name: str = "MELD"
        self.data_root: str = "working/dataset/MELD_preprocessed"
        self.data_valid: str = "val.pkl"
        self.text_max_length: int = 297
        self.audio_max_length: int = 546220

        # Config name
        self.name = (
            f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
