from abc import ABC, abstractmethod

from deeplearning.qnn_blocks import *
from deeplearning.qnn import *
from deeplearning.networks import *
from deeplearning.dataset import CustomizedDataset

class StoQNN(ABC):
    def __init__(self):
        pass

    def latent_parameters(self):
        for module in self.modules():
            yield module.parameter.latent_param

nn_registry = {
    "ternary": TernaryNeuralNet,
    "type1":  FullPrecisionNet_Type1
}
