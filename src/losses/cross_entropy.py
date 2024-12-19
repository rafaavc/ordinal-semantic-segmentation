import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from .loss import LossProvider, Loss
from models.model_output import ModelOutput


class CrossEntropy(Loss):
    def forward(self, output: ModelOutput, ytrue):
        return cross_entropy(output.get_final_output_logits(), ytrue)
    
    def convert_to_probs(self, preds=None):
        if preds is None:
            return True
        return F.softmax(preds, 1)

class CrossEntropy_LP(LossProvider):
    def create_loss(self):
        return CrossEntropy(self.num_classes)
