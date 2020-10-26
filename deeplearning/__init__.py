from abc import ABC, abstractmethod
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
