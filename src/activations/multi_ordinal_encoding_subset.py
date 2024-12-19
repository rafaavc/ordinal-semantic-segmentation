import torch
from .activation import Activation, ActivationProvider


class MultiOrdinalEncodingSubset(Activation):
    def how_many_outputs(self):
        return self.K-1

    def forward(self, x):
        """
        -> unknown (0)
            -> environment (1)
                -> road (2)
                    -> sidewalk (3)
                    -> road agents (4) [? this is questionable, agents may not be on the road (even in this more extense definition of road)
                    -> drivable area (5)
                        -> ego lane (6)

        # ordinal encoding
                        
        output = [
            P(C > 0),
                P(C > 1),
                    P(C > 2, C = 3),
                    P(C > 2, C = 4),
                    P(C > 2, C = 5),
                        P(C > 5)
        ]
        P(0) = 1 - P(C > 0)
            P(1) = P(C > 0) - P(C > 1)
                P(2) = P(C > 1) - SUM(P(C > 2, C = 3), P(C > 2, C = 4), P(C > 2, C = 5))
                    P(3) = P(C > 2, C = 3)
                    P(4) = P(C > 2, C = 4)
                    P(5) = P(C > 2, C = 5) - P(C > 5)
                        P(6) = P(C > 5)
        """
        probs = torch.sigmoid(x)
        probs = torch.cat((
            1 - probs[:, [0]],
            probs[:, [0]] - probs[:, [1]],
            probs[:, [1]] - (probs[:, [2]] + probs[:, [3]] + probs[:, [4]]),
            probs[:, [2]],
            probs[:, [3]],
            probs[:, [4]] - probs[:, [5]],
            probs[:, [5]]
        ), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return { 'probs': probs, 'before_ordinal_enc': x }


class MultiOrdinalEncodingSubset_AP(ActivationProvider):
    def create_activation(self):
        return MultiOrdinalEncodingSubset(self.num_classes)
