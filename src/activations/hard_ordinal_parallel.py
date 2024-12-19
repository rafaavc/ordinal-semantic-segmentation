import torch
from torch.nn import Module
from .activation import ActivationProvider
from .hard_ordinal import HardOrdinal


class HardOrdinalParallel(HardOrdinal):
    # This version is similar to CORAL.
    def __init__(self, K):
        super().__init__(K)
        biases = torch.empty(1, self.K-1, 1, 1)
        torch.nn.init.xavier_uniform_(biases)
        self.biases = torch.nn.parameter.Parameter(biases)

    def how_many_outputs(self):
        return 1

    def forward(self, x):
        return super().forward(x+self.biases)


class HardOrdinalParallel_AP(ActivationProvider):
    def create_activation(self):
        return HardOrdinalParallel(self.num_classes)
    