import torch.nn.functional as F
from .loss import Loss, LossProvider
from models.model_output import ModelOutput
from datasets.ordinality_tree import OrdinalityTree
from .multi_co2 import multi_neighbor_term as co2
from .multi_cs_tv_v4 import multi_neighbor_term as csnp, get_contact_surface_operations
from .multi_co2 import get_unimodal_operations


class Multi_CO2_CSNP(Loss):
    def __init__(self, K, lamda=0.01, omega=0.05, target_is_onehot=False, tree: OrdinalityTree=None):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda
        self.omega = omega

        assert tree.is_multi_ordinal, "The ordinality tree should be multi ordinal for Multi_CO2_CSNP loss. Use CO2_CSNP instead."
        self.cs_operations = get_contact_surface_operations(tree)
        self.co2_operations = get_unimodal_operations(tree)

    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)
        
        assert self.K == P.shape[1] # sanity check
        
        ce = F.cross_entropy(ypred, ytrue)
        co2_term = co2(P, self.onehot(ytrue), self.omega, self.co2_operations)
        csnp_term = csnp(P, self.cs_operations)

        # applying same lambda to both because the best Dice
        # result from each of them occur at +/- the same lambda levels
        return ce + self.lamda * (co2_term + csnp_term)
    
    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class Multi_CO2_CSNP_LP(LossProvider):
    def create_loss(self) -> Loss:
        return Multi_CO2_CSNP(self.num_classes, lamda=self.reg_weight, tree=self.tree)
