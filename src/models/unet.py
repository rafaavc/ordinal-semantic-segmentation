from .model_provider import ModelProvider
from .unet_impl import UNet
from activations import Activation
from .model_output import ModelOutput


class CustomUNet(UNet):
    def __init__(self, n_channels: int, n_outputs: int, activation: Activation, bilinear=False):
        super().__init__(n_channels, n_outputs, bilinear)
        self.activation = activation
    
    def forward(self, x):
        logits = super().forward(x) # logits
        if self.activation is None:
            return ModelOutput(final_output=logits, final_output_is_probs=False)
        
        act_out = self.activation(logits) # probabilities
        before_ordinal_encoding = act_out['before_ordinal_enc'] if 'before_ordinal_enc' in act_out else None
        return ModelOutput(final_output=act_out['probs'], final_output_is_probs=True, \
                           logits_before_activation=logits, output_before_ordinal_encoding=before_ordinal_encoding)


class UNet_MP(ModelProvider):
    def __init__(self, pretrained: bool, n_channels: int, how_many_outputs: int, activation: Activation):
        super().__init__(pretrained, n_channels, how_many_outputs, activation)
    
    def create_model(self):
        return CustomUNet(self.n_channels, self.how_many_outputs, self.activation)
