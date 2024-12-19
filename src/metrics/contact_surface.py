import torch
from sys import stderr
from torchmetrics import Metric
from typing import Optional
from datasets.ordinality_tree import OrdinalityTree
from losses.multi_cs_tv_v4 import get_contact_surface_operations


def calculate_contact_surface_multi_ordinal(probs, operations):
    dx_invalid = 0
    dx_all = 0
    dy_invalid = 0
    dy_all = 0
    for k, (k_level, ops) in operations.items():
        probs_copy = torch.ones_like(probs) * 500.
        probs_copy[probs == k] = k_level
        for kk, kk_level in ops:
            probs_copy[probs == kk] = kk_level

        dx = torch.abs(probs_copy[:,  :, :-1] - probs_copy[:, :, 1:]).type(torch.int32)
        dy = torch.abs(probs_copy[:,  :-1, :] - probs_copy[:, 1:, :]).type(torch.int32)

        dx *= dx < 200
        dy *= dy < 200
        
        dx_invalid += torch.count_nonzero((dx != 0) & (dx != 1))
        dx_all += torch.count_nonzero(dx != 0)
        dy_invalid += torch.count_nonzero((dy != 0) & (dy != 1))
        dy_all += torch.count_nonzero(dy != 0)

    return dx_invalid, dx_all, dy_invalid, dy_all


def calculate_contact_surface(probs, operations):
    if len(probs.shape) == 4:
        probs = torch.argmax(probs, dim=1)

    if operations:
        return calculate_contact_surface_multi_ordinal(probs, operations)

    dx = torch.abs(probs[:,  :, :-1] - probs[:, :, 1:]).type(torch.int32)
    dy = torch.abs(probs[:,  :-1, :] - probs[:, 1:, :]).type(torch.int32)

    # percentage of invalid ordinal jumps out of all the jumps
    return torch.count_nonzero((dx != 0) & (dx != 1)), torch.count_nonzero(dx != 0), \
        torch.count_nonzero((dy != 0) & (dy != 1)), torch.count_nonzero(dy != 0)


class OrdinalContactSurface(Metric):
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    
    def __init__(self, tree: OrdinalityTree):
        super().__init__()
        self.operations = get_contact_surface_operations(tree, log=False, level_threshold=0) if tree.is_multi_ordinal else None
        self.add_state("dx_incorrect", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dx_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dy_incorrect", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dy_total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, _: torch.Tensor):
        dx_incorrect, dx_total, dy_incorrect, dy_total = calculate_contact_surface(probs, self.operations)

        self.dx_incorrect += dx_incorrect
        self.dx_total += dx_total
        self.dy_incorrect += dy_incorrect
        self.dy_total += dy_total

    def compute(self):
        dx = self.dx_incorrect / self.dx_total if self.dx_total != 0 else 0
        dy = self.dy_incorrect / self.dy_total if self.dy_total != 0 else 0
        return (dx + dy) / 2
