import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Coordinate Attention Module
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, 1)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, 1)
        self.conv_w = nn.Conv2d(mip, in_channels, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv1(y))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()

# Hybrid Model using EfficientNet-B3 and ResNet50 with Coordinate Attention
class HybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.effnet = models.efficientnet_b3(weights="DEFAULT")
        self.resnet = models.resnet50(weights="DEFAULT")
        self.effnet.classifier = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.ca_eff = CoordinateAttention(1536)
        self.ca_res = CoordinateAttention(2048)
        self.classifier = nn.Sequential(
            nn.Linear(1536 + 2048, 1024),
            nn.LayerNorm(1024),  # Replaced BatchNorm1d with LayerNorm for small batch sizes
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # EfficientNet-B3 branch
        eff = self.ca_eff(self.effnet.features(x))
        eff = F.adaptive_avg_pool2d(eff, 1).flatten(1)
        # ResNet50 branch
        x_r = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x_r = self.resnet.layer4(
            self.resnet.layer3(
                self.resnet.layer2(
                    self.resnet.layer1(
                        self.resnet.maxpool(x_r)
                    )
                )
            )
        )
        x_r = self.ca_res(x_r)
        x_r = F.adaptive_avg_pool2d(x_r, 1).flatten(1)
        # Concatenate and classify
        return self.classifier(torch.cat([eff, x_r], dim=1))

# Optional: Focal Loss for imbalanced datasets
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()