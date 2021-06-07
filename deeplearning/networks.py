import numpy as np

# PyTorch Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn

class NaiveMLP(nn.Module):
    def __init__(self, in_dims, out_dims, dim_hidden=200):
        super(NaiveMLP, self).__init__()
        self.fc1 = nn.Linear(in_dims, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)
        
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)

        self.fc3 = nn.Linear(dim_hidden, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out

class FullPrecisionNet(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecisionNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

class LeNet_5(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(LeNet_5, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.fc_input_size, 512)
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

class VGG_7(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(VGG_7, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/8)**2 * 512
        
        self.conv1_1 = nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.bn4 = nn.BatchNorm1d(1024, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(1024, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.mp3(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_8(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(ResNet_8, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.avg_pool_input_size = int(self.input_size/8)

        self.in_planes = 128
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(BasicBlock, 128, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 256, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, out_dims)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mp1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out1 = self.fc(out)
        return out