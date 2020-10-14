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
        self._init_theta()

    def _init_theta(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        latent_param = torch.zeros_like(self.weight.data)
        latent_param = latent_param[..., None]
        latent_param = torch.repeat_interleave(latent_param, 2, dim=-1)

        latent_param.requires_grad = True
        self.weight.latent_param = latent_param.clone()

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
        self._init_theta()

    def _init_theta(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        latent_param = torch.zeros_like(self.weight.data)
        latent_param = latent_param[..., None]
        latent_param = torch.repeat_interleave(latent_param, 2, dim=-1)

        latent_param.requires_grad = True
        self.weight.latent_param = latent_param.clone()

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

