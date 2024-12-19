from .loss import Loss, LossProvider
from torch.nn.functional import binary_cross_entropy


class BinaryCrossEntropy(Loss):
    def forward(self, ypred, ytrue):
        ypred = ypred["out"]
        ytrue = self.onehot(ytrue)
        return binary_cross_entropy(ypred, ytrue)


class BinaryCrossEntropy_LP(LossProvider):
    def create_loss(self):
        return BinaryCrossEntropy(self.num_classes)
