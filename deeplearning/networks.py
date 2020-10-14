import numpy as np

# PyTorch Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn

class FullPrecisionNet_Type1(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_Type1, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.input_size, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

