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

        self.loss_type = "CrossEntropyLoss_CombinedMarginLoss_Optim"

        self.checkpoint_dir = "checkpoints_latest/IEMOCAP_losses_optim"

        self.optimizer_type: str = "Adam"  # Adam, SGD, AdamW

        self.transfer_learning = True
        self.model_type = "_4M_SER"
        self.trainer = "MarginTrainer"

        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "vggish"
        self.audio_encoder_dim = 128
        self.audio_unfreeze = True

        # For combined margin loss
        self.margin_loss_m1 = 1.0
        self.margin_loss_m2 = 0.0
        self.margin_loss_m3 = 0.5  # 1.0
        self.margin_loss_scale = 64.0
        self.fusion_dim: int = self.audio_encoder_dim

        # For contrastive-center loss
        self.lambda_c = 1.0
        self.feat_dim = self.fusion_dim
        # Dataset
        self.data_name: str = "IEMOCAP"
        self.data_root: str = "data/IEMOCAP_preprocessed"
        self.data_valid: str = "val.pkl"

        # Optim params
        self.lambda_total = [i/10 for i in range(1,10)]
        self.optim_attributes = ["lambda_total"]

        # Config name
        self.name = f"{self.model_type}_CosFace_{self.text_encoder_type}_{self.audio_encoder_type}"

        for key, value in kwargs.items():
            setattr(self, key, value)
