from .loss import Loss, LossProvider
from .dice import calc_dice


class GeometricDice(Loss):
    def forward(self, pred, target):
        pred = pred['out']
        target = self.onehot(target)

        result = 1.
        for c in range(self.K): # for each class
            result *= (0.5 + 0.5 * calc_dice(pred[:, c], target[:, c]))

        return 1. - result


class GeometricDice_LP(LossProvider):
    def create_loss(self):
        return GeometricDice(self.num_classes)
