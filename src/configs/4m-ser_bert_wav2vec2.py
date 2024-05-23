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

        self.checkpoint_dir = "checkpoints_latest/ESD"

        self.optimizer_type = "Adam"

        self.transfer_learning = False
        self.model_type = "_4M_SER"
        self.trainer = "Trainer"

        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "wav2vec2_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.linear_layer_output = [128, 64]

        self.fusion_dim: int = self.audio_encoder_dim

        # Dataset
        self.data_name: str = "ESD"
        self.data_root: str = "data/ESD_preprocessed"
        self.data_valid: str = "val.pkl"

        # Config name
        self.name = (
            f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
