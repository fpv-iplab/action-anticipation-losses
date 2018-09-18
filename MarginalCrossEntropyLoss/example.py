from MarginalCrossEntropyLoss import MarginalCrossEntropyLoss
import numpy as np
import pandas as pd
import torch

#read list of actions/verbs/nouns
actions=pd.read_csv('actions.csv',index_col='id')

def get_marginal_indexes(mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi=[]
    for v in range(actions[mode].max()+1):
        vals=actions[actions[mode]==v].index.values
        if len(vals)>0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi

#get lists of action indexes for each verb and noun
vi = get_marginal_indexes('verb')
ni = get_marginal_indexes('noun')

#number of action classes
numclass = len(actions['action'].unique())

#build loss 
loss = MarginalCrossEntropyLoss([vi,ni], numclass)

#fake scores predicted by the model
scores = torch.rand(64,numclass) #batch: 64, classes: numclass

#fake ground truth actions, verbs and nouns
y_actions = torch.rand(64,numclass).argmax(1) #gt action labels
y_verbs = torch.rand(64,len(vi)).argmax(1) #gt verb labels
y_nouns = torch.rand(64,len(ni)).argmax(1) #gt noun labels

#stack labels for verbs nouns and actions
l=loss(scores,torch.stack([y_verbs, y_nouns, y_actions],1))
print(l)
