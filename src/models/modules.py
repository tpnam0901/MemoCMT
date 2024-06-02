import torch.nn as nn
import torchaudio
from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    FocalNetConfig,
    FocalNetModel,
)

from configs.base import Config


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True
    )
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


def build_roberta_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    roberta = RobertaModel.from_pretrained("roberta-base", config=config)
    return roberta


def build_focalnet_tiny_encoder() -> nn.Module:
    """A function to build focalnet encoder"""
    config = FocalNetConfig.from_pretrained(
        "microsoft/focalnet-tiny", output_hidden_states=True
    )
    focalnet = FocalNetModel.from_pretrained("microsoft/focalnet-tiny", config=config)
    return focalnet


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    """A function to build hubert encoder"""
    return HuBertBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "focalnet_t": build_focalnet_tiny_encoder,
        "hubert_base": HuBertBase,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type]()


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
        "roberta": build_roberta_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()
