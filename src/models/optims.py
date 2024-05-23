from torch import nn, optim
from configs.base import Config
from typing import Union


def adamw(cfg: Config, network: nn.Module) -> optim.AdamW:
    return optim.AdamW(
        params=network.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=0.01,
    )


def adam(cfg: Config, network: nn.Module) -> optim.Adam:
    return optim.Adam(
        params=network.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )


def sgd(cfg: Config, network: nn.Module) -> optim.SGD:
    return optim.SGD(
        params=network.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momemtum,
        weight_decay=cfg.sdg_weight_decay,
    )


def get_optim(cfg: Config, network: nn.Module) -> Union[optim.SGD, optim.Adam]:
    optim_fn = {
        "SGD": sgd,
        "Adam": adam,
        "AdamW": adamw,
    }
    assert cfg.optimizer_type in optim_fn.keys(), (
        "Invalid optimizer_type. The valid optim is ["
        + " ".join(list(optim_fn.keys()))
        + "]"
    )

    return optim_fn[cfg.optimizer_type](cfg, network)
