import math
import abc
import numpy as np
import textwrap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models

class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(CNNEncoder, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),  # (32, 23, 23)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (64, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.AdaptiveAvgPool2d((6, 6))  # Downsample to (64, 6, 6)
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(256, output_dim)   # Map to 128 dimensions
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# TODO: reduce input size to 96*96 or 84*84

class ResNetDepthEncoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt the first conv layer for 1-channel input
        orig_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=orig_conv1.bias
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:] = orig_conv1.weight.mean(dim=1, keepdim=True)

        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # Save the original in_features before replacing fc
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Project to 128D
        self.project = nn.Linear(in_features, output_dim)

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, depth_image):
        features = self.backbone(depth_image)         # [B, 512]
        embedding = self.project(self.dropout(features))            # [B, 128]
        
        return embedding


def freeze_encoder(encoder):
    for param in encoder.backbone.parameters():
        param.requires_grad = False
