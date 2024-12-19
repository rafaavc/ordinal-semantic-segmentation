from .loss import Loss, LossProvider
from .cross_entropy_probs import CrossEntropyProbs
from .total_variation import TotalVariation
from .loss_reg_simple import LossRegSimple


class CE_TV(LossRegSimple):
    def __init__(self, K, target_is_onehot=False, lambda_tv=1):
        super().__init__(
            K,
            main_loss=CrossEntropyProbs(K),
            reg_loss=TotalVariation(K),
            lmbda=lambda_tv,
            target_is_onehot=target_is_onehot
        )



class CE_TV_LP(LossProvider):
    def create_loss(self) -> Loss:
        return CE_TV(self.num_classes, lambda_tv=self.reg_weight)

