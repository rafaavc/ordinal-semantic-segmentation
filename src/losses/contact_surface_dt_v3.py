import torch
from .loss import Loss, LossProvider
from .contact_surface_tv_v4 import ContactSurface
from kornia.contrib import distance_transform


def distance_term(P, ytrue, K):
    measure = 0 # probability difference between classes, weighted to their ordinal delta, for pixels in neighborhood
    count = 0

    min_dist = 10.
    activations = 1. * (P > .5)
    DT = distance_transform(activations)
    DT *= DT < min_dist
    DT = min_dist - DT
    DT *= DT != min_dist
    DT += min_dist * activations

    for k in range(K):
        for kk in range(k+2, K):
            if abs(kk - k) <= 1:
                continue

            ordinal_multiplier = abs(kk - k) - 1 # more weight to more ordinally distant classes

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


class ContactSurfaceDTv3_LP(LossProvider):
    def create_loss(self) -> Loss:
        return ContactSurface(self.num_classes, lamda=self.reg_weight, term=distance_term)
