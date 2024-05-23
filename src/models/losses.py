import torch
from torch.nn import CrossEntropyLoss as CELoss
from configs.base import Config


class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0]
        return super().forward(out, target)
