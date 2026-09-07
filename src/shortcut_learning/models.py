from collections.abc import Iterator

import torch
from torch import nn
from torchvision.models import resnet18


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = 128
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))

    def feature_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.features.parameters()
        yield from self.projection.parameters()

    def reset_classifier(self) -> None:
        old_weight = self.classifier.weight
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(
            device=old_weight.device,
            dtype=old_weight.dtype,
        )


class ResNet18MNIST(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))

    def feature_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.parameters()

    def reset_classifier(self) -> None:
        old_weight = self.classifier.weight
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(
            device=old_weight.device,
            dtype=old_weight.dtype,
        )


def build_model(name: str, num_classes: int = 10) -> LeNet | ResNet18MNIST:
    if name == "lenet":
        return LeNet(num_classes=num_classes)
    if name == "resnet18_mnist":
        return ResNet18MNIST(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
