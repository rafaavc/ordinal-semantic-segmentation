import torch


def onehot(x, k):
    return torch.moveaxis(torch.nn.functional.one_hot(x, k), -1, 1).float()


class Loss(torch.nn.Module):
    def __init__(self, K, target_is_onehot = False):
        super().__init__()
        self.K = K
        self.target_is_onehot = target_is_onehot

    def onehot(self, target):
        if not self.target_is_onehot:
            return onehot(target, self.K)
        return target

    def convert_to_probs(self, preds=None):
        return False


class LossProvider:
    def __init__(self, num_classes: int, reg_weight: float, tree):
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.tree = tree

    def create_loss(self) -> Loss:
        pass
