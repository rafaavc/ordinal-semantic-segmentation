import torch
import torch.nn.functional as F
from datasets.ordinality_tree import OrdinalityTree
from .loss import Loss, LossProvider
from models.model_output import ModelOutput
from .multi_cs_tv_v4 import get_contact_surface_operations
from kornia.contrib import distance_transform


def multi_neighbor_term(P, operations):
    measure = 0 # probability difference between classes, weighted to their ordinal delta, for pixels in neighborhood
    count = 0
    min_dist = 10.

    activations = 1. * (P > .5)
    DT = distance_transform(activations)
    DT *= DT < min_dist
    DT = min_dist - DT
    DT *= DT != min_dist
    DT += min_dist * activations

    for k in operations.keys():
        k_level, ops = operations[k]
        for kk, kk_level in ops:
            if abs(kk_level - k_level) <= 1:
                print("Found one op that isn't valid")
                continue

            ordinal_multiplier = abs(kk_level - k_level) - 1 # more weight to more ordinally distant classes

            d_k, d_kk = DT[:, k], DT[:, kk]
            p_k, p_kk = P[:, k], P[:, kk]

            calc = p_k * d_kk + p_kk * d_k
            calc = calc[calc != 0]

            if calc.shape[0] == 0: # this is safe because calc is always >= 0
                continue

            measure += ordinal_multiplier * torch.mean(calc)
            count += 1
    
    if count != 0:
        measure /= count
    
    measure /= min_dist
    return measure


class MultiContactSurfaceDTv3(Loss):
    def __init__(self, K, lamda=0.01, target_is_onehot=False, tree: OrdinalityTree=None):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda

        assert tree.is_multi_ordinal, "The ordinality tree should be multi ordinal for MultiCO2 loss. Use CO2 instead."
        self.operations = get_contact_surface_operations(tree)        


    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)

        term = multi_neighbor_term(P, self.operations)
        ce = 0 if ytrue is None else F.cross_entropy(ypred, ytrue) # for semi supervised learning

        return ce + self.lamda * term
    

    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class MultiContactSurfaceDTv3_LP(LossProvider):
    def create_loss(self) -> Loss:
        return MultiContactSurfaceDTv3(self.num_classes, lamda=self.reg_weight, tree=self.tree)
