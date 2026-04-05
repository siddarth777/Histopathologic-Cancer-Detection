import torch.nn as nn
import torchvision.models as models


class BaseCNN(nn.Module):
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_alexnet(num_classes=1):
    model = models.alexnet(weights=None)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model


def build_resnet50(num_classes=1):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vgg16(num_classes=1):
    model = models.vgg16(weights=None)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model


MODEL_REGISTRY = {
    'cnn': BaseCNN,
    'alexnet': build_alexnet,
    'resnet50': build_resnet50,
    'vgg16': build_vgg16,
}


def get_model(name: str):
    builder = MODEL_REGISTRY[name]
    return builder() if name == 'cnn' else builder(num_classes=1)
