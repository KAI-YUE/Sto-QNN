import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal

#--------------------------------
# Ternary neural network blocks following the paper 
# Shayer, Oran, Dan Levi, and Ethan Fetaya. "Learning discrete weights using the local reparameterization trick." 
# ICLR 2018.
#---------------------------------

class TernaryConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernaryConv2d, self).__init__(*kargs, **kwargs)
        self.latentdim = 3
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        weight_shape = torch.tensor(self.weight.shape).tolist()
        weight_shape.append(3)
        self.weight.latent_param = torch.zeros(weight_shape, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        theta = torch.sigmoid(self.weight.latent_param)
        ones_tensor = torch.ones_like(theta[..., 0])
        theta_ = ones_tensor - theta[..., 0] - theta[..., 1]
        mu = theta[..., 1]*ones_tensor  - theta_*ones_tensor
        sigma_square = theta[..., 1]*ones_tensor  + theta_*ones_tensor

        mean = nn.functional.conv2d(input, mu, self.bias, 
                                    self.stride, self.padding, self.dilation)
        sigma_square = nn.functional.conv2d(input**2, sigma_square, None, 
                                    self.stride, self.padding, self.dilation)

        epsilon = torch.randn_like(mean)
        out = mean + sigma_square.sqrt()*epsilon

        return out

class TernaryLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(TernaryLinear, self).__init__(*kargs, **kwargs)
        self.latentdim = 3
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        weight_shape = torch.tensor(self.weight.shape).tolist()
        weight_shape.append(3)
        self.weight.latent_param = torch.zeros(weight_shape, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        theta = torch.sigmoid(self.weight.latent_param)
        ones_tensor = torch.ones_like(theta[..., 0])
        theta_ = ones_tensor - theta[..., 0] - theta[..., 1]
        mu = theta[..., 1]*ones_tensor  - theta_*ones_tensor
        sigma_square = theta[..., 1]*ones_tensor  + theta_*ones_tensor

        mean = nn.functional.linear(input, mu, self.bias)
        sigma_square = nn.functional.linear(input**2, sigma_square)

        epsilon = torch.randn_like(mean)
        out = mean + sigma_square.sqrt()*epsilon

        return out

