import torch
import torch.nn.functional as F
from datasets.ordinality_tree import OrdinalityTree, OrdinalityTreeNode, TreeLevelNodeTracker
from .loss import Loss, LossProvider
from models.model_output import ModelOutput


def multi_neighbor_term(P, operations):
    measure = 0 # probability difference between classes, weighted to their ordinal delta, for pixels in neighborhood
    count = 0

    for k in operations.keys():
        k_level, ops = operations[k]
        for kk, kk_level in ops:
            if abs(kk_level - k_level) <= 1:
                print("Found one op that isn't valid")
                continue

            ordinal_multiplier = abs(kk_level - k_level) - 1 # more weight to more ordinally distant classes

            dx = P[:,  k, :, :-1] * P[:, kk, :, 1:]
            dy = P[:,  k, :-1, :] * P[:, kk, 1:, :]

            if dx.shape[0] == 0 or dy.shape[0] == 0:
                continue

            measure += ordinal_multiplier * (torch.mean(dx) + torch.mean(dy)) / 2
            count += 1

    if count != 0:
        measure /= count
    return measure


def get_contact_surface_operations(tree: OrdinalityTree, log: bool = True, level_threshold=1):
    assert not tree.contains_abstract_nodes, "MultiCS only supports ordinality trees without abstract nodes."
    if log:
        print(tree)

    operations = {}

    def assemble_operations(current_node: OrdinalityTreeNode, ascendants: TreeLevelNodeTracker, children_output, current_level: int):
        ops = []
        for level, asc in ascendants.dict.items():
            ops += [ (ascendant.value, level) for ascendant in asc if abs(current_level - level) > level_threshold ]

        descendants_tracker = TreeLevelNodeTracker()
        for child_desc_tracker in children_output:
            descendants_tracker.merge(child_desc_tracker)
            for level, desc in child_desc_tracker.dict.items():
                ops += [ (descendant.value, level) for descendant in desc if abs(level - current_level) > level_threshold ]

        operations[current_node.value] = (current_level, ops)

        descendants_tracker.add(current_node, current_level)
        return descendants_tracker
    
    tree.visit(assemble_operations)
    
    if log:
        print(get_contact_surface_operations_str(operations))

    return operations


def get_contact_surface_operations_str(ops):
    result = ""
    all_ops = sorted([ (k, ops) for k, ops in ops.items() ])
    for k, ops in all_ops:
        result += f"{k}: {ops}\n"
    return result


class MultiContactSurfaceTVv4(Loss):
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


class MultiContactSurfaceTVv4_LP(LossProvider):
    def create_loss(self) -> Loss:
        return MultiContactSurfaceTVv4(self.num_classes, lamda=self.reg_weight, tree=self.tree)
