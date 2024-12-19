import torch.nn.functional as F
from .loss import Loss, LossProvider
from models.model_output import ModelOutput
from .co2 import neighbor_term as co2
from .contact_surface_tv_v4 import neighbor_term_v4 as csnp


class CO2_CSNP(Loss):
    def __init__(self, K, lamda=0.01, omega=0.05, target_is_onehot=False):
        super().__init__(K, target_is_onehot)
        self.lamda = lamda
        self.omega = omega

    def forward(self, output: ModelOutput, ytrue):
        ypred = output.get_final_output_logits()
        P = output.get_probs(self)
        
        assert self.K == P.shape[1] # sanity check
        
        ce = F.cross_entropy(ypred, ytrue)
        co2_term = co2(P, ytrue, self.omega)
        csnp_term = csnp(P, self.onehot(ytrue), self.K)

        # applying same lambda to both because the best Dice
        # result from each of them occur at +/- the same lambda levels
        return ce + self.lamda * (co2_term + csnp_term)
    
    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)


class CO2_CSNP_LP(LossProvider):
    def create_loss(self) -> Loss:
        return CO2_CSNP(self.num_classes, lamda=self.reg_weight)
