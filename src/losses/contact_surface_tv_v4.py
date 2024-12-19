import torch
import torch.nn.functional as F
from .loss import Loss, LossProvider
from models.model_output import ModelOutput


def neighbor_term_v4(P, ytrue, K):
    measure = 0
    count = 0

    for k in range(K):
        for kk in range(K):
            if abs(kk - k) <= 1:
                continue

            ordinal_multiplier = abs(kk - k) - 1 # more weight to more ordinally distant classes

            dx = P[:,  k, :, :-1] * P[:, kk, :, 1:]
            dy = P[:,  k, :-1, :] * P[:, kk, 1:, :]

            measure += ordinal_multiplier * (torch.mean(dx) + torch.mean(dy)) / 2
            count += 1
    
    if count != 0:
        measure /= count
    return measure


class ContactSurface(Loss):
    def __init__(self, K, lamda=0.1, omega=0.05, target_is_onehot=False, term=None):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda
        self.omega = omega
        self.term = term

    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)
        
        assert self.K == P.shape[1] # sanity check

        term = self.term(P, self.onehot(ytrue), self.K)
        ce = F.cross_entropy(ypred, ytrue)

        return ce + self.lamda * term
    
    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class ContactSurfaceTVv4_LP(LossProvider):
    def create_loss(self) -> Loss:
        return ContactSurface(self.num_classes, lamda=self.reg_weight, term=neighbor_term_v4)
