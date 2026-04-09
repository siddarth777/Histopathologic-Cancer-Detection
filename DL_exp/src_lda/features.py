from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.models import BaseCNN


class BaseCNNBackbone(nn.Module):
    output_dim = 256

    def __init__(self):
        super().__init__()
        model = BaseCNN()
        self.features = model.features
        self.pool = model.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class AlexNetBackbone(nn.Module):
    output_dim = 4096

    def __init__(self):
        super().__init__()
        model = tv_models.alexnet(weights=None)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet50Backbone(nn.Module):
    output_dim = 2048

    def __init__(self):
        super().__init__()
        model = tv_models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class VGG16Backbone(nn.Module):
    output_dim = 4096

    def __init__(self):
        super().__init__()
        model = tv_models.vgg16(weights=None)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


BACKBONE_REGISTRY = {
    'cnn': BaseCNNBackbone,
    'alexnet': AlexNetBackbone,
    'resnet50': ResNet50Backbone,
    'vgg16': VGG16Backbone,
}


def build_backbone(name: str) -> nn.Module:
    return BACKBONE_REGISTRY[name]()
