from torch import nn, Tensor
from typing import Dict
from .model_provider import ModelProvider
from torchvision.models.segmentation import DeepLabV3
from activations import Activation
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead


def create_deeplabv3(pretrained: bool, num_classes: int) -> DeepLabV3:
    if pretrained:
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=21, aux_loss=True)
        model.classifier = DeepLabHead(2048, num_classes)
        model.aux_classifier = FCNHead(1024, num_classes)
        return model
    return deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)


class CustomDeepLabV3(nn.Module):
    def __init__(self, deeplabv3: DeepLabV3, num_classes: int, activation: Activation):
        super().__init__()
        self.deeplabv3 = deeplabv3
        self.num_classes = num_classes
        self.activation = activation

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        logits = self.deeplabv3(x)['out']
        if self.activation is None:
            return { 'out': logits, 'logits': logits }
        
        probs = self.activation(logits)
        return { 'out': probs }


class DeepLabV3_MP(ModelProvider):
    def __init__(self, pretrained: bool, n_channels: int, how_many_outputs: int, activation: Activation):
        super().__init__(pretrained, n_channels, how_many_outputs, activation)
    
    def create_model(self) -> CustomDeepLabV3:
        return CustomDeepLabV3(create_deeplabv3(self.pretrained, self.how_many_outputs), self.how_many_outputs, self.activation)
