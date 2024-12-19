import torch
from torch.nn.functional import nll_loss
from .loss import LossProvider, Loss
from models.model_output import ModelOutput


class CrossEntropyProbs(Loss):
    def forward(self, output: ModelOutput, ytrue, eps=1e-7):
        ypred = output.get_final_output_probs()
        ypred = torch.log(ypred + eps)
        return nll_loss(ypred, ytrue)


class CrossEntropyProbs_LP(LossProvider):
    def create_loss(self):
        return CrossEntropyProbs(self.num_classes)
