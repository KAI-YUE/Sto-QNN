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
        # theta = torch.sigmoid(self.weight.latent_param)
        theta = torch.sigmoid(self.weight.latent_param)
        mu = theta[..., 1]  - theta[..., 0]
        sigma_square = theta[..., 1]  + theta[..., 0] - mu**2

        mu = F.conv2d(input, mu, self.bias, 
                      self.stride, self.padding, self.dilation)
        sigma_square = F.conv2d(input**2, sigma_square, None, 
                                self.stride, self.padding, self.dilation)

        # to prevent sqrt(x) yields inf grad at 0, filter out zero entries

        epsilon = torch.randn_like(mu)
        out = mu + (sigma_square + 0.1).sqrt()*epsilon

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
        weight_shape.append(2)
        self.weight.latent_param = torch.zeros(weight_shape, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        # theta = torch.sigmoid(self.weight.latent_param)
        theta = torch.sigmoid(self.weight.latent_param)
        mu = theta[..., 1]  - theta[..., 0]
        sigma_square = theta[..., 1]  + theta[..., 0] - mu**2

        mu = F.linear(input, mu, self.bias)
        sigma_square = F.linear(input**2, sigma_square)

        # to prevent sqrt(x) yields inf grad at 0, filter out zero entries

        epsilon = torch.randn_like(mu)
        out = mu + (sigma_square + 0.1).sqrt()*epsilon

        return out

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinaryConv2d, self).__init__(*kargs, **kwargs)
        self.latentdim = 2
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.quant_alpha = 0.2
    
    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        self.weight.latent_param = torch.zeros_like(self.weight, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        # theta = activate_fun(w) = 1/2*tanh(w) + 1/2
        theta = torch.tanh(self.quant_alpha*self.weight.latent_param)
        # mu = 2*theta - 1
        # sigma_square = 1 - mu**2

        # mu = F.conv2d(input, mu, self.bias,
        #               self.stride, self.padding, self.dilation)
        # sigma_square = F.conv2d(input**2+1e-6, sigma_square, None, 
        #               self.stride, self.padding, self.dilation)
        
        # epsilon = torch.randn_like(mu)
        # out = mu + (sigma_square + 0.1).sqrt()*epsilon

        mu = F.conv2d(input, theta, self.bias,
                      self.stride, self.padding, self.dilation)
        out = mu

        return out

class BinaryLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinear, self).__init__(*kargs, **kwargs)
        self.latentdim = 2
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.quant_alpha = 0.2

    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        self.weight.latent_param = torch.zeros_like(self.weight, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        # theta = activate_fun(w) 
        theta = torch.tanh(self.quant_alpha*self.weight.latent_param)
        # mu = 2*theta - 1
        # sigma_square = 1 - mu**2

        # mu = F.linear(input, mu, self.bias)
        # sigma_square = F.linear(input**2, sigma_square, None)
        
        # epsilon = torch.randn_like(mu)
        # out = mu + (sigma_square + 0.1).sqrt()*epsilon

        mu = F.linear(input, theta, self.bias)
        out = mu

        return out