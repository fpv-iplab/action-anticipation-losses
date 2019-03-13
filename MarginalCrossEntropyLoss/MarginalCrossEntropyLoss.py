from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class MarginalCrossEntropyLoss(_Loss):
    """Implements the "Marginal Cross Entropy Loss" from the paper:
        A. Furnari, S. Battiato, G. M. Farinella (2018). 
        Leveraging Uncertainty to Rethink Loss Functions and Evaluation Measures for Egocentric Action Anticipation . 
        In International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV ."
    """
    def __init__(self, marginal_indexes, numclass, reduction='mean'):
        """Marginal Cross Entropy Loss
        Input:
            marginal_indexes: list of indexes for each of the marginal probabilities
            numclass: number of classes (e.g., number of actions)
            size_average: whether to average the losses in the batch (if False, they are summed)
            reduce: if False, all individual losses are applied, otherwise they are summed/averaged depending on size_average
            """
        super().__init__(reduction=reduction)
        marginal_masks = []
        for mi in marginal_indexes:
            masks = []
            for i in mi:
                masks.append(self.__indexes_to_masks(i, numclass))
            marginal_masks.append(torch.stack(masks))

        self.marginal_masks = marginal_masks

    def __indexes_to_masks(self, indexes, maxlen):
        mask = np.zeros(maxlen)
        for i in indexes:
            mask[i] = 1
        return torch.from_numpy(mask)

    def __sum_exps(self, exps):
        return exps.sum(1)

    def __build_marginal_loss(self, input, marginal_target, marginal_masks):
        mask = torch.Tensor(marginal_masks[marginal_target.cpu().data].float())
        mask = mask.to(input.device)
        exps = torch.exp(input)
        sum_all = self.__sum_exps(exps)
        sum_marginal = exps.mul(mask).sum(1)
        return torch.log(sum_all) - torch.log(sum_marginal)

    def forward(self, input, marginal_targets):
        """  input: predicted scores
             marginal_targets: list of targets of the marginal probabilities"""
        #input: bs x nc
        #marginal_targets: list of marginal targets
        loss = torch.Tensor(torch.zeros(input.shape[0])).to(input.device)
        for i, mm in enumerate(self.marginal_masks):
            l = self.__build_marginal_loss(input, marginal_targets[:, i], mm)
            loss += l

        #sum cross entropy on actions
        loss += F.cross_entropy(input, marginal_targets[:, -1], reduction='none')

        if self.reduction is not None:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss
