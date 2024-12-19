import torch
from torch.nn import Module
from .activation import ActivationProvider
from .multi_hard_ordinal import MultiHardOrdinal


class MultiHardOrdinalParallel(MultiHardOrdinal):
    # This version is similar to CORAL.
    def __init__(self, K, tree):
        super().__init__(K, tree=tree)
        biases = torch.empty(1, self.K-1, 1, 1)
        torch.nn.init.xavier_uniform_(biases)
        self.biases = torch.nn.parameter.Parameter(biases)

    # def how_many_outputs(self):
    #     return 1

    def forward(self, x):
        return super().forward(x+self.biases)


class MultiHardOrdinalParallel_AP(ActivationProvider):
    def create_activation(self):
        return MultiHardOrdinalParallel(self.num_classes, self.tree)
    