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


class BinaryNeuralNet(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = BinaryConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = BinaryConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = BinaryLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        # x = torch.relu(self.conv1(x))
        x = self.mp1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        # x = torch.relu(self.conv2(x))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.bn3(self.fc1(x)))
        # x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

    def freeze_norm_layers(self):
        self.bn1.weight.requires_grad = False
        self.bn1.bias.requires_grad = False

        self.bn2.weight.requires_grad = False
        self.bn2.bias.requires_grad = False

        self.bn3.weight.requires_grad = False
        self.bn3.bias.requires_grad = False

class BinaryNeuralNet_Type2(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(BinaryNeuralNet_Type2, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv_block1 = nn.Sequential(
            BinaryConv2d(self.in_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(),
            BinaryConv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.conv_block2 = nn.Sequential(
            BinaryConv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            # nn.Tanh(),
            nn.ReLU(),
            BinaryConv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.conv_block3 = nn.Sequential(
            BinaryConv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = BinaryLinear(512, 256)
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
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class CompleteTernaryNeuralNet(StoQNN):
    def __init__(self, in_dims, in_channels, out_dims=10):  
        super(CompleteTernaryNeuralNet, self).__init__()
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

# initialization method reference 
# Roth, Wolfgang, Günther Schindler, Holger Fröning, and Franz Pernkopf. 
# "Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions." 
# In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 382-398. Springer, Cham, 2019.

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
    if "method" in kwargs:
        method = kwargs["method"]
    else:
        method = "probability" 
    

    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)

    if method == "probability":
        delta = p_max - p_min/2
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            
            # pr(w_q = -1)
            ref_w = ref_state_dict[module_name + ".weight"]
            ref_w = 0.8*ref_w/ref_w.std()

            idx = torch.logical_and(ref_w >-1, ref_w<=0)
            prob_m1 = torch.where(ref_w <= -1, p_max, ref_w)
            prob_m1 = torch.where(ref_w > 0, p_min/2, prob_m1)
            prob_m1[idx] = p_max - delta*(ref_w[idx] + 1)

            # pr(w_q = 1)
            idx = torch.logical_and(ref_w >0, ref_w<=1)
            prob_p1 = torch.where(ref_w > 1, p_max, ref_w)
            prob_p1 = torch.where(ref_w <= 0, p_min/2, prob_p1)
            prob_p1[idx] = p_min/2 + delta*ref_w[idx]

            # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
            module.weight.latent_param.data[..., 0] = torch.log(-1+1/(1-prob_m1))
            module.weight.latent_param.data[..., 1] = torch.log(-1+1/(1-prob_p1))

    elif method == "shayer":
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
            normalized_w = ref_w.clamp(-0.99, 0.99)

            # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
            prob_m1 = p_min - p_max*normalized_w
            prob_m1 = torch.clamp(prob_m1, p_min, p_max)
            prob_p1 = p_min  + p_max*normalized_w
            prob_p1 = torch.clamp(prob_p1 , p_min, p_max)

            module.weight.latent_param.data[..., 0] = torch.log(-1+1/(1-prob_m1))
            module.weight.latent_param.data[..., 1] = torch.log(-1+1/(1-prob_p1))


def init_bnn_params(model, ref_model, **kwargs):
    """Initialize the Bernoulli distribution parameters. 
    Pr(w_b=1) = 0.5*(1 + w) 

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
    if "method" in kwargs:
        method = kwargs["method"]
    else:
        # method = "probability"
        method = "plain"
        # method = "test" 

    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)

    if method == "probability":
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
                
            # normalize the weight
            ref_w = ref_state_dict[module_name + ".weight"]
            # normalized_w = ref_w.clamp(-0.99, 0.99)    # restrict the weight to (-1,1)
            # abs_normalized_w = normalized_w.abs()

            idx = torch.logical_and(ref_w>-1, ref_w<1)
            prob_p1 = torch.where(ref_w <= -1, p_min/2, ref_w)
            prob_p1 = torch.where(ref_w > 1, p_max, prob_p1)
            prob_p1[idx] = p_min/2 + 0.5*(p_max - p_min/2)*(ref_w[idx] + 1) 
            
            module.weight.data = torch.log(-1 + 1/(1 - prob_p1))
    
    elif method == "plain":
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
                
            # normalize the weight
            ref_w = ref_state_dict[module_name + ".weight"]
            normalized_w = ref_w.clamp(-0.99, 0.99)    # restrict the weight to (-1,1)
            # abs_normalized_w = normalized_w.abs()

            module.weight.data = torch.log(-1 + 2/(1 - normalized_w))

    else:
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            ref_w = ref_state_dict[module_name + ".weight"]
            module.weight.data = torch.where(ref_w>0, torch.tensor(7.), torch.tensor(-7.))