from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn as nn

class StoQNN(ABC, nn.Module):
    def __init__(self):
        super(StoQNN, self).__init__()

    def latent_parameters(self):
        for module in self.modules():
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            yield module.weight.latent_param

    def latent_param_dict(self):
        lparam_dict = OrderedDict()
        named_modules = self.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue

            lparam_dict[module_name + ".weight.latent_param"] = module.weight.latent_param

        return lparam_dict

    def load_latent_param_dict(self, lparam_dict):
        for latent_param_name, latent_param in lparam_dict.items():
            exec("self.{:s} = latent_param".format(latent_param_name))

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

from deeplearning.qnn_blocks import *
from deeplearning.qnn import *
from deeplearning.qnn_gn import *
from deeplearning.networks import *
from deeplearning.dataset import CustomizedDataset

nn_registry = {
    "ternary":          TernaryNeuralNet,
    "tgn1":             TernaryNeuralNet_GN_Type1,
    "tgn2":             TernaryNeuralNet_GN_Type2,

    "binary1":          BinaryNeuralNet_Type1,
    "binary2":          BinaryNeuralNet_Type2,
    "binary":           BinaryNeuralNet_noBN,

    "binary_complete":  BinaryNeuralNet_Complete,

    "ternary_complete": TernaryNeuralNet_Complete,
    
    "bn1":            FullPrecisionNet_BN1,
    "bn2":            FullPrecisionNet_BN2,
    "gn1":              FullPrecisionNet_GN_Type1,
    "gn2":              FullPrecisionNet_GN_Type2 
}
