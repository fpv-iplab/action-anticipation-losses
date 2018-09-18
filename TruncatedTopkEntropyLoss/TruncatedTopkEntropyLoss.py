from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss, _assert_no_grad
from torch.nn import functional as F

class TruncatedTopkEntropyLoss(_Loss):
    """Implements the Truncated TOP-K Entropy Loss from the paper
        
        Lapin, Maksim, Matthias Hein, and Bernt Schiele. 
        "Loss functions for top-k error: Analysis and insights." 
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
    """
    def __init__(self, k, size_average=True, reduce=True):
        super(TruncatedTopkEntropyLoss, self).__init__(size_average=size_average)
        assert(k>1) #for k=1 use standard cross entropy
        self.k = k
        self.reduce=reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        fy = input[range(target.size()[0]),target]
        fj = input

        aj = fj - fy.view(-1,1)

        if self.k!=1:
            aj[range(target.size()[0]),target]=np.inf #set positions where the target is to infinity to avoid select them among the smalles m-k components
            aj=aj.topk(aj.shape[1]-self.k,largest=False)[0] #select the smallest m-k components
        
        loss = torch.log(1+torch.exp(aj).sum(1))

        if self.reduce:
            if self.size_average:
                loss=loss.mean()
            else:
                loss=loss.sum()

        return loss

