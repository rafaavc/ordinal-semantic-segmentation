import torch
from .activation import Activation, ActivationProvider


################################################################################
# Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network        #
# approach to ordinal regression." 2008 IEEE international joint conference on #
# neural networks (IEEE world congress on computational intelligence). IEEE,   #
# 2008. https://arxiv.org/pdf/0704.1028.pdf                              #
################################################################################
# Notice that some authors cite later papers like OR-CNN (Zhenxing Niu et al,  #
# 2016) but we believe this was the first for neural networks and is based on  #
# the Frank & Hall (2001) ordinal ensemble.                                    #
################################################################################

class OrdinalEncoding(Activation):
    def how_many_outputs(self):
        return self.K-1

    def forward(self, x):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(x)
        prob_0 = 1-probs[:, [0]]
        prob_k = probs[:, [-1]]
        probs = torch.cat((prob_0, probs[:, :-1]-probs[:, 1:], prob_k), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return { 'probs': probs, 'before_ordinal_enc': x }


class OrdinalEncoding_AP(ActivationProvider):
    def create_activation(self):
        return OrdinalEncoding(self.num_classes)
