from typing import Optional
import torch, json, math
from losses.loss import onehot
from losses.multi_co2 import get_unimodal_operations
from torchmetrics import Metric

from matplotlib import pyplot as plt


def is_unimodal_segmentation_multi_ordinal(P, operations):
    K = P.shape[1]
    P_mask = onehot(torch.argmax(P, dim=1), K)

    res = 0
    batch_size = 30
    for i in range(math.ceil(P.shape[0] / batch_size)):
        batch_left = i*batch_size
        batch_right = min((i+1)*batch_size, P.shape[0])
        
        P_mask_batch = P_mask[batch_left:batch_right]
        P_batch = P[batch_left:batch_right]
        result = torch.ones(P_batch.shape[0], 1, P_batch.shape[2], P_batch.shape[3], device=P.device) == 1

        for k in range(K):
            if k not in operations or torch.count_nonzero(P_mask_batch[:, k]) == 0:
                continue
            # execute operations for when k is the correct class

            k_mask = P_mask_batch[:, [k]] == 1
            operation = operations[k]

            left_op = operation[0]
            if len(left_op) > 1:
                k_mask_op = k_mask.repeat(1, len(left_op)-1, 1, 1)
                select = torch.diff(P_batch[:, left_op, :, :], dim=1) * k_mask_op
                result = result & (torch.sum(select >= 0, dim=1) == select.shape[1])[:, None]

            if len(operation) > 1:
                for right_op in operation[1:]:
                    k_mask_op = k_mask.repeat(1, len(right_op)-1, 1, 1)
                    select = torch.diff(P_batch[:, right_op, :, :], dim=1) * k_mask_op
                    result = result & (torch.sum(select <= 0, dim=1) == select.shape[1])[:, None]
        
        res += len(result[result == 1])
    return res


def is_unimodal_segmentation(p, _):
    if len(p.shape) == 4: # N, K, H, W
        p = torch.reshape(p, (p.shape[0], p.shape[1], p.shape[2]*p.shape[3]))
    
    assert len(p.shape) == 3, "for metric is_unimodal_segmentation preds need to be in shape (N, K, H*W)"
    
    zero = torch.zeros((p.shape[0], 1, p.shape[2]), device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero, dim=1), decimals=2))
    p = torch.diff(p, dim=1)
    p = torch.count_nonzero(p, dim=1)
    return len(p[p == 1])


class UnimodalPercentage(Metric):
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    
    def __init__(self, tree):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        
        if tree.is_multi_ordinal:
            print("Calculating unimodal percentage for multi ordinal tree.")
            self.unimodal_operations = get_unimodal_operations(tree)
            self.is_unimodal = is_unimodal_segmentation_multi_ordinal
        else:
            self.unimodal_operations = None
            self.is_unimodal = is_unimodal_segmentation

    def update(self, ppred: torch.Tensor, _: torch.Tensor):
        total = ppred.shape[0] * ppred.shape[2] * ppred.shape[3] # total pixels
        correct = self.is_unimodal(ppred, self.unimodal_operations)

        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct / self.total
