from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class TruncatedTopkEntropyLoss(_Loss):
    """Implements the Truncated TOP-K Entropy Loss from the paper
        
        Lapin, Maksim, Matthias Hein, and Bernt Schiele. 
        "Loss functions for top-k error: Analysis and insights." 
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
    """

    def __init__(self, k, reduction='mean'):
        super().__init__(reduction=reduction)
        assert(k > 1)  # for k=1 use standard cross entropy
        self.k = k

    def forward(self, input, target):
        fy = input[range(target.size()[0]), target]
        fj = input

        aj = fj - fy.view(-1, 1)

        if self.k != 1:
            aj[range(target.size()[0]), target] = np.inf  # set positions where the target is to infinity to avoid select them among the smalles m-k components
            aj = aj.topk(aj.shape[1] - self.k, largest=False)[0]  # select the smallest m-k components

        loss = torch.log(1 + torch.exp(aj).sum(1))

        if self.reduction is not None:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss
