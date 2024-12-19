import torch
from .activation import Activation, ActivationProvider

class Softmax(Activation):
    def forward(self, x):
        probs = torch.nn.functional.softmax(x, 1)
        return { 'probs': probs }
    

class Softmax_AP(ActivationProvider):
    def create_activation(self):
        return Softmax(self.num_classes)
