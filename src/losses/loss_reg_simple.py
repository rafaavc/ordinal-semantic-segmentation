from .loss import Loss


class LossRegSimple(Loss):
    def __init__(self, K, main_loss, reg_loss, lmbda, target_is_onehot=False):
        super().__init__(K, target_is_onehot)

        self.main_loss = main_loss
        self.regularization_term = reg_loss
        self.lmbda = lmbda

    def forward(self, pred, target):
        main_loss = self.main_loss(pred, target)
        reg_loss = self.regularization_term(pred)
        return main_loss + self.lmbda * reg_loss

