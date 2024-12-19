import torch
from .loss import Loss
from typing import Optional, List


class TotalVariation(Loss):
    def __init__(self, K, target_is_onehot=False, classes: Optional[List[int]] = None):
        super().__init__(K, target_is_onehot)

        assert type(classes) in [list, type(None)], f"[TotalVariation] invalid 'classes': {classes}"
        print(f"Classes for total variation: {classes}")

        self.classes = classes # classes to apply it to - if none, all of them
        self.showed_msg = False

    def forward(self, pred):
        before_ordinal_enc = 'before_ordinal_enc' in pred and pred['before_ordinal_enc'] is not None
        if before_ordinal_enc and not self.showed_msg:
            print("Calculating total variation with logits before ordinal encoding")
            self.showed_msg = True  
        
        pred = pred['before_ordinal_enc'] if before_ordinal_enc else pred['out']
        
        # calculate the horizontal and vertical gradients
        if self.classes is None: # apply to all classes
            dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        else:
            dx = torch.abs(pred[:, self.classes, :, :-1] - pred[:, self.classes, :, 1:])
            dy = torch.abs(pred[:, self.classes, :-1, :] - pred[:, self.classes, 1:, :])
        
        # sum the gradient values and return the mean
        return torch.mean(dx) + torch.mean(dy)
