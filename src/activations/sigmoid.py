import torch
from .activation import Activation, ActivationProvider

class Sigmoid(Activation):
    def forward(self, x):
        return { 'probs': torch.sigmoid(x) }
    

class Sigmoid_AP(ActivationProvider):
    def create_activation(self):
        return Sigmoid(self.num_classes)
