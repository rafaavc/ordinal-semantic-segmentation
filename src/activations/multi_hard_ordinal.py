import torch
from .activation import Activation, ActivationProvider
from .multi_ordinal_encoding import encode


def convert_dfs(probs, cumprod, current_class_idx, children, cat_list):
    cumprod = torch.mul(cumprod, probs[:, [current_class_idx-1]])
    cat_list.append(cumprod)

    if children is not None:
        for child in sorted([k for k in children.keys() if k != 'is_abstract']):
            convert_dfs(probs, cumprod, child, children[child], cat_list)
    

def convert_from_conditional(probs, tree):
    cat_list = []
    convert_dfs(probs, torch.ones_like(probs[:, [0]]), 1, tree[0][1], cat_list)
    return torch.cat(cat_list, 1)


class MultiHardOrdinal(Activation):

    # the version from Kelwin paper is slightly different than CORN: there is no
    # subset, and the probabilities are < (inferior), not > (superior).
    def how_many_outputs(self):
        return self.K-1

    def forward(self, x):
        # model returns conditional "corrected" probabilities
        # convert the conditional "corrected" probabilities to "corrected" probabilities
        probs = torch.sigmoid(x)

        # probs = torch.cat((
        #     torch.cumprod(probs[:, :3], 1),
        #     torch.cumprod(torch.cat((probs[:, :2], probs[:, [3]]), 1), 1)[:, [-1]],
        #     torch.cumprod(torch.cat((probs[:, :2], probs[:, 4:]), 1), 1)[:, -2:],
        # ), 1)
        probs = convert_from_conditional(probs, self.tree)
        assert probs.shape[1] == self.how_many_outputs()

        after_conditional = probs

        # convert the "corrected" probabilities to probabilities
        # probs = torch.cat((
        #     1 - probs[:, [0]],
        #     probs[:, [0]] - probs[:, [1]],
        #     probs[:, [1]] - (probs[:, [2]] + probs[:, [3]] + probs[:, [4]]),
        #     probs[:, [2]],
        #     probs[:, [3]],
        #     probs[:, [4]] - probs[:, [5]],
        #     probs[:, [5]]
        # ), 1)
        probs = encode(probs, self.tree)

        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return { 'probs': probs, 'before_ordinal_enc': after_conditional }


class MultiHardOrdinal_AP(ActivationProvider):
    def create_activation(self):
        return MultiHardOrdinal(self.num_classes, self.tree)
