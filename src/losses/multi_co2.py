import torch
from .loss import Loss, LossProvider
import torch.nn.functional as F
from datasets.ordinality_tree import OrdinalityTree, OrdinalityTreeNode, TreeLevelNodeTracker
from models.model_output import ModelOutput


def multi_neighbor_term(P, ytrue, margin, operations):
    margin = torch.tensor(margin, device=ytrue.device)
    K = P.shape[1]

    loss = 0
    for k in range(K):
        if torch.count_nonzero(ytrue[:, k]) == 0:
            continue

        # execute operations for when k is the correct class

        k_mask = ytrue[:, [k]] == 1
        operation = operations[k]

        results = []

        left_op = operation[0]
        if len(left_op) > 1:
            k_mask_op = k_mask.repeat(1, len(left_op)-1, 1, 1)
            results.append(F.relu(margin - torch.diff(P[:, left_op, :, :], dim=1)[k_mask_op]))

        if len(operation) > 1:
            for right_op in operation[1:]:
                k_mask_op = k_mask.repeat(1, len(right_op)-1, 1, 1)
                results.append(F.relu(margin + torch.diff(P[:, right_op, :, :], dim=1)[k_mask_op]))
        
        tmp = torch.cat(results, 0)
        loss += torch.mean(tmp)
    
    loss /= K
    return loss


def get_unimodal_operations(tree: OrdinalityTree):
    assert not tree.contains_abstract_nodes, "MultiCO2 only supports ordinality trees without abstract nodes."
    print(tree)

    operations = {}

    def assemble_operations(current_node: OrdinalityTreeNode, ascendants: TreeLevelNodeTracker, children_output, current_level: int):
        ops = [[]]
        for level in ascendants.dict.values():
            ops[-1] += [ ascendant.value for ascendant in level ]
        ops[-1].append(current_node.value)

        for child in current_node.children:
            ops.append([ current_node.value, child.value ])
        
        for output in children_output:
            ops += [ [*op] for op in output ]

        operations[current_node.value] = ops
        return ops[1:]
    
    tree.visit(assemble_operations)

    print(get_unimodal_operations_str(operations))

    return operations


def get_unimodal_operations_str(ops):
    result = ""
    all_ops = sorted([ (k, ops) for k, ops in ops.items() ])
    for k, ops in all_ops:
        result += f"{k}: {ops}\n"
    return result


class MultiCO2(Loss):
    def __init__(self, K, lamda=0.01, omega=0.05, target_is_onehot=False, tree: OrdinalityTree=None):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda
        self.omega = omega

        assert tree.is_multi_ordinal, "The ordinality tree should be multi ordinal for MultiCO2 loss. Use CO2 instead."
        self.operations = get_unimodal_operations(tree)        


    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)
        
        term = multi_neighbor_term(P, self.onehot(ytrue), self.omega, self.operations)
        ce = F.cross_entropy(ypred, ytrue)

        return ce + self.lamda * term
    

    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class MultiCO2_LP(LossProvider):
    def create_loss(self) -> Loss:
        return MultiCO2(self.num_classes, lamda=self.reg_weight, tree=self.tree)
