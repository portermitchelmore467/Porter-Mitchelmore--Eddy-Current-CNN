import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpedanceCNN(nn.Module):
    """Simple configurable CNN for impedance-plane images.

    Input: (B, C=1, H, W)
    Output: logits of shape (B, num_classes) or a single regression value (num_classes=1)
    """

    def __init__(self, in_channels=1, num_classes=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.conv1 = nn.Conv2d(in_channels, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f * 4, f * 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(f * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = self.pool(x)
        x = F.relu(self.conv4(x)); x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_model(**kwargs):
    return ImpedanceCNN(**kwargs)
