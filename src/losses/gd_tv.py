from .loss import Loss, LossProvider
from .geometric_dice import GeometricDice
from .total_variation import TotalVariation
from .loss_reg_simple import LossRegSimple


class GD_TV(LossRegSimple):
    def __init__(self, K, target_is_onehot=False, lambda_tv=1):
        super().__init__(
            K,
            main_loss=GeometricDice(K),
            reg_loss=TotalVariation(K),
            lmbda=lambda_tv,
            target_is_onehot=target_is_onehot
        )


class GD_TV_LP(LossProvider):
    def create_loss(self) -> Loss:
        return GD_TV(self.num_classes, lambda_tv=self.reg_weight)

