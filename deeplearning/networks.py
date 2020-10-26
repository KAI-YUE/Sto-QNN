import numpy as np

# PyTorch Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn

class NaiveCNN(nn.Module):
    def __init__(self, channels=1, dim_out=10, **kwargs):
        super(NaiveCNN, self).__init__()
        self.channels = channels
        if "dim_in" in kwargs:
            self.input_size = int(np.sqrt(kwargs["dim_in"]/channels))
        else:
            self.input_size = 28
        
        kernel_size = 3
        self.fc_input_size = (((((self.input_size - kernel_size)/1 + 1) - kernel_size)/1 + 1) - kernel_size)/2 + 1
        self.fc_input_size = int(self.fc_input_size)**2 * 20

        self.predictor = nn.Sequential(
                    nn.Conv2d(channels, 10, kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size=kernel_size),
                    nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                    nn.ReLU(),
                    )
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], self.channels, self.input_size, self.input_size)
        x = self.predictor(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FullPrecisionNet_Type1(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_Type1, self).__init__()
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
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = self.mp1(x)
        
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.mp2(x)

        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class FullPrecisionNet_Type2(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet_Type2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.fc1 = nn.Linear(512, 256)
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        x = self.drop(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x