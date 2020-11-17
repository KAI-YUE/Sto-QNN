import numpy as np

# PyTorch Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn

class FullPrecisionNet_GN_Type1(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_GN_Type1, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
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

class FullPrecisionNet_GN_Type2(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_GN_Type2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
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
        # x = F.relu(self.gn3(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class FullPrecisionNet_BN1(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_BN1, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
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

class FullPrecisionNet_BN2(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_BN2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512)
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