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
        
        self.conv1 = TernaryConv2d(self.in_channels, 32, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TernaryLinear(1024, 512)
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

class BinaryNeuralNet(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 32, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BinaryConv2d(32, 64, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = BinaryLinear(1024, 512)
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


class CompleteTernaryNeuralNet(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(CompleteTernaryNeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = TernaryConv2d(self.in_channels, 32, kernel_size=5)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=5)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TernaryLinear(1024, 512)
        self.fc2 = TernaryLinear(512, out_dims)

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

def init_latent_params(model, ref_model, **kwargs):
    """Initialize the multinomial distribution parameters. 
    theta_0 = Pr(w=-1) 
    theta_1 = Pr(w=1)

    Args:
        ref_model (nn.Module):     the reference floating point model
    """
    if "p_max" in kwargs:
        p_max = torch.tensor(kwargs["p_max"])
    else:
        p_max = torch.tensor(0.95)
    if "p_min" in kwargs:
        p_min = torch.tensor(kwargs["p_min"])
    else:
        p_min = torch.tensor(0.05)

    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)
    for module_name, module in named_modules:
        if not hasattr(module, "weight"):
            continue
        elif not hasattr(module.weight, "latent_param"):
            continue
            
        # normalize the weight
        ref_w = ref_state_dict[module_name + ".weight"]
        # normalized_w = ref_w / ref_w.std()
        # abs_normalized_w = normalized_w.abs()
        # normalized_w = 2*(ref_w - ref_w.min())/(ref_w.max() - ref_w.min()) - 1
        normalized_w = ref_w

        # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
        prob_minus_one = p_min - p_max*normalized_w
        prob_minus_one = torch.clamp(prob_minus_one, p_min, p_max)
        prob_one = p_min  + p_max*normalized_w
        prob_one = torch.clamp(prob_one , p_min, p_max)

        module.weight.latent_param.data[..., 0] = torch.log(-1+1/(1-prob_minus_one))
        module.weight.latent_param.data[..., 1] = torch.log(-1+1/(1-prob_one))


def init_bnn_params(model, ref_model):
    """Initialize the Bernoulli distribution parameters. 
    Pr(w_b=1) = 0.5*(1 + w) 

    Args:
        ref_model (nn.Module):     the reference floating point model
    """
    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)
    for module_name, module in named_modules:
        if not hasattr(module, "weight"):
            continue
        elif not hasattr(module.weight, "latent_param"):
            continue
            
        # normalize the weight
        ref_w = ref_state_dict[module_name + ".weight"]
        normalized_w = ref_w.clamp(-0.99, 0.99)    # restrict the weight to (-1,1)
        # abs_normalized_w = normalized_w.abs()

        module.weight.data = torch.log(-1 + 2/(1-normalized_w))