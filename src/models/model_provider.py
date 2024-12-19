from activations import Activation


class ModelProvider():
    def __init__(self, pretrained: bool, n_channels: int, how_many_outputs: int, activation: Activation):
        self.pretrained = pretrained
        self.how_many_outputs = how_many_outputs
        self.activation = activation
        self.n_channels = n_channels

    def create_model(self):
        pass
