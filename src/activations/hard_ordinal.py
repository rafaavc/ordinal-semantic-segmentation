import torch
from .activation import Activation, ActivationProvider


class HardOrdinal(Activation):
    # the version from Kelwin paper is slightly different than CORN: there is no
    # subset, and the probabilities are < (inferior), not > (superior).
    def how_many_outputs(self):
        return self.K-1

    def forward(self, x):
        # model returns conditional "corrected" probabilities
        # convert the conditional "corrected" probabilities to "corrected" probabilities
        sigmoids = torch.sigmoid(x)
        probs_plus = torch.cumprod(sigmoids, 1)
        # convert the "corrected" probabilities to probabilities
        after_conditional = probs_plus
        prob_0 = 1-probs_plus[:, [0]]
        prob_k = probs_plus[:, [-1]]
        probs = torch.cat((prob_0, probs_plus[:, :-1]-probs_plus[:, 1:], prob_k), 1)
        # there may be small discrepancies
        # probs = torch.clamp(probs, 0, 1)
        # probs = probs / probs.sum(1, keepdim=True)
        return { 'probs': probs, 'before_ordinal_enc': after_conditional }


class HardOrdinal_AP(ActivationProvider):
    def create_activation(self):
        return HardOrdinal(self.num_classes)
