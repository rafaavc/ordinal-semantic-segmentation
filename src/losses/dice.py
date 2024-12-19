import torch
from .loss import Loss, LossProvider

def calc_dice(pred, target, smooth = 1.):
        """
        https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
        This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        
        return (2. * intersection + smooth) / (A_sum + B_sum + smooth)


class Dice(Loss):
    def forward(self, pred, target):
        pred = pred['out']
        target = self.onehot(target)
        result = 0
        for c in range(self.K): # for each class
            result += 1 - calc_dice(pred[:, c], target[:, c])
        return result / self.K


class Dice_LP(LossProvider):
    def create_loss(self):
        return Dice(self.num_classes)
