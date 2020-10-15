import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from deeplearning import StoQNN
from deeplearning.qnn_blocks import TernaryConv2d, TernaryLinear

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

def init_latent_params(model, ref_model, **kwargs):
    """Initialize the multinomial distribution parameters. 
    theta_0 = Pr(w=0) = sigmoid(a) = p_max - (p_max - p_min)|w|, 
    theta_1 = Pr(w=1) = sigmoid(b) = 0.5*(1 + |w|/(1-Pr(w=0))) * Pr(w!=0) = 0.5*((1-theta_0) + |w|)

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
        normalized_w = ref_w
        abs_normalized_w = normalized_w.abs()

        # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
        # a, b denote the variables which controls the probability
        sigmoid_a = p_max - (p_max - p_min) * abs_normalized_w
        sigmoid_a = torch.clamp(sigmoid_a, p_min, p_max)
        sigmoid_b = 0.5*((1 - sigmoid_a) + normalized_w)
        sigmoid_b = torch.clamp(sigmoid_b, p_min, p_max)

        module.weight.latent_param.data[..., 0] = torch.log(-1 + 1/(1 - sigmoid_a))
        module.weight.latent_param.data[..., 1] = torch.log(-1 + 1/(1 - sigmoid_b))




