import torch


class Activation(torch.nn.Module):
    def __init__(self, K, tree=None):
        super().__init__()
        self.K = K
        self.tree = tree

    def how_many_outputs(self):
        return self.K


class ActivationProvider:
    def __init__(self, num_classes, tree):
        self.num_classes = num_classes
        self.tree = tree

    def create_activation(self) -> Activation:
        pass
