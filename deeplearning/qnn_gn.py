import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from deeplearning import StoQNN
from deeplearning.qnn_blocks import *

class TernaryNeuralNet_GN_Type1(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(TernaryNeuralNet_GN_Type1, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = TernaryConv2d(self.in_channels, 64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=64, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = TernaryConv2d(64, 128, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = TernaryLinear(2048, 512)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=512, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.gn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.gn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class TernaryNeuralNet_GN_Type2(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(TernaryNeuralNet_GN_Type2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = TernaryConv2d(self.in_channels, 64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = TernaryConv2d(64, 128, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = TernaryLinear(2048, 512)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.gn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.gn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

    def freeze_norm_layers(self):
        self.gn1.weight.requires_grad = False
        self.gn1.bias.requires_grad = False
        self.gn2.weight.requires_grad = False
        self.gn2.bias.requires_grad = False
        self.gn3.weight.requires_grad = False
        self.gn3.bias.requires_grad = False