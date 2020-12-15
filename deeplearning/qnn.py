import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from deeplearning import StoQNN
from deeplearning.qnn_blocks import *

class TernaryNeuralNet(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(TernaryNeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = TernaryConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = TernaryConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = TernaryLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class BinaryNeuralNet_noBN(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet_noBN, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 64, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = BinaryConv2d(64, 128, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = BinaryLinear(2048, 512)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        
        x = torch.relu(self.conv1(x))
        x = self.mp1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class BinaryNeuralNet_Type1(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet_Type1, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = BinaryConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = BinaryLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class BinaryNeuralNet_Complete(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet_Complete, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = BinaryConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = BinaryLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        # self.fc2 = BinaryLinear(512, out_dims)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False


class BinaryNeuralNet_Type2(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet_Type2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = BinaryConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = BinaryLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False


class TernaryNeuralNet_Complete(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):  
        super(TernaryNeuralNet_Complete, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = TernaryConv2d(self.in_channels, 64, kernel_size=5)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = TernaryConv2d(64, 128, kernel_size=5)
        # self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = TernaryLinear(2048, 512)
        # self.bn3 = nn.BatchNorm1d(512, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = TernaryLinear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False
