from typing import Any
from torch import Tensor
from torchmetrics import MeanAbsoluteError
from losses.loss import onehot

class MeanAbsoluteErrorCustom(MeanAbsoluteError):
    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.K = num_classes

    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds, onehot(target, self.K))
