import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.delta = 1e-8

    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        weight_shape = torch.tensor(self.weight.shape).tolist()
        weight_shape.append(2)
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
        mu = theta[..., 1]  - theta_
        sigma_square = theta[..., 1]  + theta_ - mu**2

        mean = F.conv2d(input, mu, self.bias, 
                        self.stride, self.padding, self.dilation)
        sigma_square = F.conv2d(input**2, sigma_square, None, 
                                self.stride, self.padding, self.dilation)

        # to prevent sqrt(x) yields inf grad at 0, filter out zero entries
        non_zero_indices = (sigma_square != 0)

        # non_zero_numbers = torch.sum(non_zero_indices).to(torch.float32)
        # total_numbers = torch.prod(torch.tensor(sigma_square.shape))
        # print("fraction: {:.2f}".format(non_zero_numbers/total_numbers))

        sigma = torch.zeros_like(sigma_square)
        sigma[non_zero_indices] = sigma[non_zero_indices].sqrt()

        epsilon = torch.randn_like(mean)
        out = mean + sigma*epsilon

        return out

class TernaryLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(TernaryLinear, self).__init__(*kargs, **kwargs)
        self.latentdim = 3
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.delta = 1e-8

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
        mu = theta[..., 1]  - theta_
        sigma_square = theta[..., 1]  + theta_ - mu**2

        mean = F.linear(input, mu, self.bias)
        sigma_square = F.linear(input**2, sigma_square)

        # to prevent sqrt(x) yields inf grad at 0, filter out zero entries
        non_zero_indices = (sigma_square != 0)
        sigma = torch.zeros_like(sigma_square)
        sigma[non_zero_indices] = sigma[non_zero_indices].sqrt()

        epsilon = torch.randn_like(mean)
        out = mean + sigma*epsilon

        return out

