import torch
from .activation import Activation, ActivationProvider


def dfs(probs, current_class_idx, children, cat_list):
    if current_class_idx == 0:
        cat_list.append(1 - probs[:, [0]])
    else:
        prob_greater_than_previous = probs[:, [current_class_idx-1]]
        prob_greater_than_current = 0 if children is None else \
            torch.sum(probs[:, [k - 1 for k in children.keys() if k != 'is_abstract']], 1, keepdim=True)
        
        cat_list.append(prob_greater_than_previous - prob_greater_than_current)
    
    if children is None: return
    for child in sorted([k for k in children.keys() if k != 'is_abstract']):
        dfs(probs, child, children[child], cat_list)


def encode(probs: torch.Tensor, tree: dict):
    cat_list = []
    dfs(probs, 0, tree[0], cat_list)
    # print(len(cat_list))
    return torch.cat(cat_list, 1)


class MultiOrdinalEncoding(Activation):
    def how_many_outputs(self):
        return self.K-1

    def forward(self, x):
        """
        -> unknown (0)
            -> environment (1)
                -> road (2)
                    -> sidewalk (3)
                    -> road agents (4) [? this is questionable, agents may not be on the road (even in this more extense definition of road)
                        -> human (5)
                            -> person (6)
                            -> rider (7)
                        -> two-wheel (8)
                            -> motorcycle (9)
                            -> bicycle (10)
                        -> other (11)
                            -> car (12)
                            -> truck (13)
                            -> bus (14)
                            -> train (15)
                    -> drivable area (16)
                        -> ego lane (17)

        # ordinal encoding
                        
        output = [
            P(C > 0),                                   0
                P(C > 1),                               1
                    P(C > 2, C = 3),                    2
                    P(C > 2, C = 4),                    3
                        P(C > 4, C = 5),                4
                            P(C > 5, C = 6),            5
                            P(C > 5, C = 7),            6
                        P(C > 4, C = 8),                7
                            P(C > 8, C = 9),            8
                            P(C > 8, C = 10),           9
                        P(C > 4, C = 11),               10
                            P(C > 11, C = 12),          11
                            P(C > 11, C = 13),          12
                            P(C > 11, C = 14),          13
                            P(C > 11, C = 15),          14
                    P(C > 2, C = 16),                   15
                        P(C > 16)                       16
        ]
        P(0) = 1 - P(C > 0)
            P(1) = P(C > 0) - P(C > 1)
                P(2) = P(C > 1) - SUM(P(C > 2, C = 3), P(C > 2, C = 4), P(C > 2, C = 16))
                    P(3) = P(C > 2, C = 3)
                    P(4) = P(C > 2, C = 4) - SUM(P(C > 4, C = 5), P(C > 4, C = 8), P(C > 4, C = 11))
                        P(5) = P(C > 4, C = 5) - SUM(P(C > 5, C = 6), P(C > 5, C = 7))
                            P(6) = P(C > 5, C = 6)
                            P(7) = P(C > 5, C = 7)
                        P(8) = P(C > 4, C = 8) - SUM(P(C > 8, C = 9), P(C > 8, C = 10))
                            P(9) = P(C > 8, C = 9)
                            P(10) = P(C > 8, C = 10)
                        P(11) = P(C > 4, C = 11) - SUM(P(C > 11, C = 12), P(C > 11, C = 13), P(C > 11, C = 14), P(C > 11, C = 15))
                            P(12) = P(C > 11, C = 12)
                            P(13) = P(C > 11, C = 13)
                            P(14) = P(C > 11, C = 14)
                            P(15) = P(C > 11, C = 15)
                    P(16) = P(C > 2, C = 16) - P(C > 16)
                        P(17) = P(C > 16)
        """
        probs = torch.sigmoid(x)
        # probs = torch.cat((
        #     1 - probs[:, [0]],
        #         probs[:, [0]] - probs[:, [1]],
        #             probs[:, [1]] - (probs[:, [2]] + probs[:, [3]] + probs[:, [15]]),
        #                 probs[:, [2]],
        #                 probs[:, [3]] - (probs[:, [4]] + probs[:, [7]] + probs[:, [10]]),
        #                     probs[:, [4]] - (probs[:, [5]] + probs[:, [6]]),
        #                         probs[:, [5]],
        #                         probs[:, [6]],
        #                     probs[:, [7]] - (probs[:, [8]] + probs[:, [9]]),
        #                         probs[:, [8]],
        #                         probs[:, [9]],
        #                     probs[:, [10]] - (probs[:, [11]] + probs[:, [12]] + probs[:, [13]] + probs[:, [14]]),
        #                         probs[:, [11]],
        #                         probs[:, [12]],
        #                         probs[:, [13]],
        #                         probs[:, [14]],
        #                 probs[:, [15]] - probs[:, [16]],
        #                     probs[:, [16]]
        # ), 1)
        probs = encode(probs, self.tree)
        # print(probs.shape)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)

        return { 'probs': probs, 'before_ordinal_enc': x }


class MultiOrdinalEncoding_AP(ActivationProvider):
    def create_activation(self):
        return MultiOrdinalEncoding(self.num_classes, self.tree)
