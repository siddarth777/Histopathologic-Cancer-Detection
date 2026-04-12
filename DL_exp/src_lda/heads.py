
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class CNNMLPNet(nn.Module):
    def __init__(self, backbone, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.backbone = backbone
        self.head = MLPHead(feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
