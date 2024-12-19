import torch
from .loss import Loss, LossProvider
from models.model_output import ModelOutput
import torch.nn.functional as F

"""
Albuquerque T, Cruz R, Cardoso J. 2021. Ordinal losses for classification of cervical cancer risk.
PeerJ Comput. Sci. 7:e457 http://doi.org/10.7717/peerj-cs.457

https://github.com/tomealbuquerque/ordinal-losses/blob/6cca7e4aabb81e228649700b211101d762f11280/mymodels.py#L152
https://github.com/rpmcruz/deep-ordinal/blob/0bb561f588e9ef9111f334b139e77d7649c27470/src/deep_ordinal/deep_ordinal.py#L751
"""


def neighbor_term_paper(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, dim=1)
    K = P.shape[1]

    loss = 0
    for k in range(K-1):
        # force previous probability to be superior to next
        reg_gt = (k < ytrue).float() * F.relu(margin+P[:,  k]-P[:, k+1])  # left
        reg_lt = (k >= ytrue).float() * F.relu(margin+P[:, k+1]-P[:, k])  # right
        loss += torch.mean(reg_gt + reg_lt)

    return loss


def neighbor_term(P, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    K = P.shape[1]

    dP = torch.diff(P, dim=1)

    sign = (torch.arange(K-1, device=ytrue.device)[None, None, None] >= ytrue[:, :, :, None])*2-1
    sign = torch.permute(sign, (0, 3, 1, 2))

    return torch.mean(F.relu(margin + sign*dP))


class CO2(Loss):
    def __init__(self, K, lamda=0.01, omega=0.05, target_is_onehot=False):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda
        self.omega = omega

    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)
        
        ce = F.cross_entropy(ypred, ytrue)
        term = neighbor_term(P, ytrue, self.omega)

        return ce + self.lamda * term
    
    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class CO2_LP(LossProvider):
    def create_loss(self) -> Loss:
        return CO2(self.num_classes, lamda=self.reg_weight)
