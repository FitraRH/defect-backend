# models/hrnet_model.py
"""
Enhanced HRNet model implementation for defect classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HRNetDefectClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(HRNetDefectClassifier, self).__init__()
        
        # Stem network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Transition layers
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage 2
        self.stage2 = nn.ModuleList([
            self._make_layer(BasicBlock, 48, 48, 4),
            self._make_layer(BasicBlock, 96, 96, 4)
        ])
        
        # Fusion modules
        self.fuse_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(96, 48, kernel_size=1, bias=False),
                    nn.BatchNorm2d(48),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            ]),
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(96)
                ),
                nn.Identity()
            ])
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))
        
        # Stage 2
        for i in range(2):
            x_list[i] = self.stage2[i](x_list[i])
        
        # Fusion
        x_fuse = []
        for i in range(2):
            y = x_list[0] if i == 0 else self.fuse_layers[i][0](x_list[0])
            for j in range(1, 2):
                if i == j:
                    y = y + x_list[j]
                else:
                    y = y + self.fuse_layers[i][j](x_list[j])
            x_fuse.append(self.relu(y))
        
        # Classification
        output = self.classifier(x_fuse[0])
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        
        return output


def create_hrnet_model(num_classes=6):
    """Create HRNet model for defect classification"""
    return HRNetDefectClassifier(num_classes=num_classes)