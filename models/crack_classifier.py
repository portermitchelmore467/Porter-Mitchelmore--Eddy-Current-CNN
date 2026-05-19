# crack_classifier.py

import torch
import torch.nn as nn
import torchvision.models as models

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, in_channels=1, use_cracked=False):
        super().__init__()
        self.use_cracked = use_cracked

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        if self.use_cracked:
            # Note: CrackedFunction is not currently implemented.!!!!!!!
            # This flag is reserved for future use.
            pass

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if self.use_cracked:
            x = self.cracked(x)
        x = self.classifier(x)
        return x

def build_model(arch="small_cnn", num_classes=2, pretrained=True, dropout=0.2, grayscale=True, use_cracked=False):
    if arch == "small_cnn":
        in_ch = 1 if grayscale else 3
        return SmallCNN(num_classes=num_classes, dropout=dropout, in_channels=in_ch, use_cracked=use_cracked)

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        if grayscale:
            w = m.conv1.weight.data.mean(dim=1, keepdim=True)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            m.conv1.weight.data = w
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        return m

    raise ValueError(f"Unknown architecture: {arch}")
