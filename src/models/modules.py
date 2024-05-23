import math
import torch
import torch.nn as nn
import torchaudio
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

from configs.base import Config
from torchvggish import vggish

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


class VGGish(nn.Module):
    def __init__(self, postprocess):
        super(VGGish, self).__init__()
        self.vggish = vggish(postprocess)

    def forward(self, x):
        out = []
        for i in range(x.size(0)):
            out.append(self.vggish(x[i]))
        x = torch.stack(out, dim=0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x

def build_vggish_encoder(cfg: Config) -> nn.Module:
    """A function to build vggish encoder"""
    return VGGish(cfg.audio_postprocess)

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


class Wav2Vec2Base(nn.Module):
    def __init__(self, **kwargs):
        super(Wav2Vec2Base, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_wav2vec2_base_encoder(cfg: Config) -> nn.Module:
    return Wav2Vec2Base()


class WavlmBase(nn.Module):
    def __init__(self, **kwargs):
        super(WavlmBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.WAVLM_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_wavlm_base_encoder(cfg: Config) -> nn.Module:
    return WavlmBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "vggish": build_vggish_encoder,
        "hubert_base": build_hubert_base_encoder,
        "wav2vec2_base": build_wav2vec2_base_encoder,
        "wavlm_base": build_wavlm_base_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)


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
