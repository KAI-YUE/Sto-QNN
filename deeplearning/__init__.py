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
        latent_params = OrderedDict()
        named_modules = self.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue

            latent_params[module_name + ".latent_param"] = module.weight.latent_param

        return latent_params

    def load_latent_param_dict(self, latent_params):
        for latent_param_name, latent_param in latent_params.items():
            exec("self.{:s} = latent_param".format(latent_param_name))

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

from deeplearning.qnn_blocks import *
from deeplearning.qnn import *
from deeplearning.networks import *
from deeplearning.dataset import CustomizedDataset

nn_registry = {
    "ternary":          TernaryNeuralNet,
    "binary":           BinaryNeuralNet,
    "binary2":          BinaryNeuralNet_Type2,
    "complete_ternary": CompleteTernaryNeuralNet,
    "complete_binary":  CompleteBinaryNeuralNet,
    "type1":            FullPrecisionNet_Type1,
    "type2":            FullPrecisionNet_Type2
}
