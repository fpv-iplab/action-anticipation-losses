from TruncatedTopkEntropyLoss import TruncatedTopkEntropyLoss
import torch

loss = TruncatedTopkEntropyLoss(5)

scores = torch.rand(64,100) #batch: 64, classes: 100
y = torch.rand(64,100).argmax(1) #gt labels

l=loss(scores,y)
print(l)

