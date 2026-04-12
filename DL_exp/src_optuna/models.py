from __future__ import annotations

import torch.nn as nn

from ..src_lda.features import build_backbone
from ..src_lda.heads import CNNMLPNet

from .config import CFG


def get_model(model_name: str) -> nn.Module:
    backbone = build_backbone(model_name)
    feature_dim = getattr(backbone, 'output_dim', None)
    if feature_dim is None:
        raise ValueError(f'Model backbone {model_name} is missing output_dim')

    return CNNMLPNet(
        backbone=backbone,
        feature_dim=int(feature_dim),
        hidden_dim=int(CFG['mlp_hidden_dim']),
        dropout=float(CFG['mlp_dropout']),
    )
