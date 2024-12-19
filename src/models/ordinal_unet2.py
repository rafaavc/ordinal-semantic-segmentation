from torch import nn
from activations import Activation
from .unet_impl.unet_parts import *
from .model_provider import ModelProvider


class OrdinalUNet2(nn.Module):
    def __init__(self, n_channels, n_outputs, activation: Activation, bilinear=False, filter_factor=32):
        super(OrdinalUNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_outputs
        self.bilinear = bilinear
        self.activation = activation

        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, filter_factor * 2 ** 0))
        self.down1 = (Down(filter_factor * 2 ** 0, filter_factor * 2 ** 1))
        self.down2 = (Down(filter_factor * 2 ** 1, filter_factor * 2 ** 2 // factor))
        
        self.up1 = (Up(filter_factor * 2 ** 2, filter_factor * 2 ** 1 // factor, bilinear))

        self.up = nn.ModuleList()
        for _ in range(n_outputs):
            self.up.append(nn.ModuleList([
                (Up(filter_factor * 2 ** 1, filter_factor * 2 ** 0, bilinear)),
                (OutConv(filter_factor * 2 ** 0, 1)),
                # (nn.Sigmoid())
            ]))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        x_up = self.up1(x3, x2)

        logits = []
        for (up2, outc) in self.up:
            x = up2(x_up, x1)
            # x = up3(x, x2)
            # x = up4(x, x1)
            lgts = outc(x)
            logits.append(lgts)
        
        logits = torch.cat(logits, dim=1)
        probs = self.activation(logits)
        return { 'out': probs, 'logits': logits }


class OrdinalUNet2_MP(ModelProvider):
    def __init__(self, pretrained: bool, n_channels: int, how_many_outputs: int, activation: Activation):
        super().__init__(pretrained, n_channels, how_many_outputs, activation)
    
    def create_model(self):
        return OrdinalUNet2(self.n_channels, self.how_many_outputs, self.activation)
