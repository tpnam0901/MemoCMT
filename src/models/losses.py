import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss as CELoss
from torch.nn.functional import linear, normalize
from configs.base import Config
from typing import Tuple, Any


class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0]
        return super().forward(out, target)


class CenterLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CenterLoss, self).__init__()

        self.feat_dim = cfg.feat_dim
        self.num_classes = cfg.num_classes
        self.lambda_c = cfg.lambda_c
        self.weight = nn.Parameter(torch.randn(cfg.num_classes, cfg.feat_dim))

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size()[0]
        expanded_centers = self.weight.index_select(dim=0, index=label)
        intra_distances = feat.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(ContrastiveCenterLoss, self).__init__()
        self.feat_dim = cfg.feat_dim
        self.num_classes = cfg.num_classes
        self.lambda_c = cfg.lambda_c
        self.weight = nn.Parameter(torch.randn(cfg.num_classes, cfg.feat_dim))

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size()[0]
        expanded_centers = self.weight.expand(batch_size, -1, -1)
        expanded_feat = feat.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_feat - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, label.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (
            (self.lambda_c / 2.0 / batch_size)
            * intra_distances
            / (inter_distances + epsilon)
            / 0.1
        )

        return loss


class CrossEntropyLoss_ContrastiveCenterLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_ContrastiveCenterLoss, self).__init__()
        self.cc_loss = ContrastiveCenterLoss(cfg)
        self.ce_loss = CELoss()

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.cc_loss(feat_fusion, label)
        total_loss = ce_loss + cc_loss
        return total_loss, logits


class CrossEntropyLoss_ContrastiveCenterLoss_Optim(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_ContrastiveCenterLoss_Optim, self).__init__()
        self.cc_loss = ContrastiveCenterLoss(cfg)
        self.ce_loss = CELoss()
        self.lambda_ = cfg.lambda_total

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.cc_loss(feat_fusion, label)
        total_loss = self.lambda_ * ce_loss + (1 - self.lambda_) * cc_loss
        return total_loss, logits


class Weighted_CrossEntropyLoss_ContrastiveCenterLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(Weighted_CrossEntropyLoss_ContrastiveCenterLoss, self).__init__()
        self.cc_loss = ContrastiveCenterLoss(cfg)
        self.ce_loss = CELoss()
        self.alpha_1 = nn.Parameter(torch.ones(1))
        self.alpha_2 = nn.Parameter(torch.ones(1))

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.cc_loss(feat_fusion, label)
        total_loss = F.relu(self.alpha_1) * ce_loss + F.relu(self.alpha_2) * cc_loss
        return total_loss, logits


class CrossEntropyLoss_CenterLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_CenterLoss, self).__init__()
        self.c_loss = CenterLoss(cfg)
        self.ce_loss = CELoss()

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        c_loss = self.c_loss(feat_fusion, label)
        total_loss = ce_loss + c_loss
        return total_loss, logits


class CrossEntropyLoss_CenterLoss_Optim(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_CenterLoss_Optim, self).__init__()
        self.c_loss = CenterLoss(cfg)
        self.ce_loss = CELoss()
        self.lambda_ = cfg.lambda_total

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        c_loss = self.c_loss(feat_fusion, label)
        total_loss = self.lambda_ * ce_loss + (1 - self.lambda_) * c_loss
        return total_loss, logits


class ContrastiveCenterLossSER(ContrastiveCenterLoss):
    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        feat_fusion = feat[1]
        loss = super().forward(feat_fusion, label)
        return loss


class CenterLossSER(CenterLoss):
    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        feat_fusion = feat[1]
        loss = super().forward(feat_fusion, label)
        return loss


class CombinedMarginLoss(nn.Module):
    def __init__(self, cfg: Config):
        """Combined margin loss for SphereFace, CosFace, ArcFace"""
        super(CombinedMarginLoss, self).__init__()
        self.in_features = cfg.feat_dim
        self.out_features = cfg.num_classes
        self.s = cfg.margin_loss_scale  # s (float): scale factor
        self.m1 = cfg.margin_loss_m1  # m1 (float): margin for SphereFace
        self.m2 = (
            cfg.margin_loss_m2
        )  # m2 (float): margin for ArcFace, m1 must be 1.0 and m3 must be 0.0
        self.m3 = (
            cfg.margin_loss_m3
        )  # m3 (float): margin for CosFace, m1 must be 1.0 and m2 must be 0.0

        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.out_features, self.in_features))
        )

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

        # CrossEntropyLoss
        self.ce_loss = CELoss()

    def forward(
        self, embbedings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[Any, torch.Tensor]:
        weight = self.weight
        norm_embeddings = normalize(embbedings)
        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        index_positive = torch.where(labels != -1)[0]
        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = (
                    final_target_logit
                )
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise ValueError("Unsupported margin values.")

        loss = self.ce_loss(logits, labels)
        return loss, logits


class CrossEntropyLoss_CombinedMarginLoss_Optim(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_CombinedMarginLoss_Optim, self).__init__()
        self.cml_loss = CombinedMarginLoss(cfg)
        self.ce_loss = CELoss()
        self.lambda_ = cfg.lambda_total

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cml_loss = self.cml_loss(feat_fusion, label)[0]
        total_loss = self.lambda_ * ce_loss + (1 - self.lambda_) * cml_loss
        return total_loss, logits


class CrossEntropyLoss_CombinedMarginLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_CombinedMarginLoss, self).__init__()
        self.cml_loss = CombinedMarginLoss(cfg)
        self.ce_loss = CELoss()

    def forward(
        self, feat: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cml_loss = self.cml_loss(feat_fusion, label)[0]
        total_loss = ce_loss + cml_loss
        return total_loss, logits


class FocalLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(FocalLoss, self).__init__()
        self.gamma = cfg.focal_loss_gamma
        self.alpha = cfg.focal_loss_alpha
        self.size_average = cfg.focal_loss_size_average
        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])
        if isinstance(self.alpha, list):
            self.alpha = torch.Tensor(self.alpha)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input[0]
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
