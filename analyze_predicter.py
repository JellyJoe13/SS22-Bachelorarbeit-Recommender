import torch_geometric.seed

"""
High level idea: execute experimental run with seed set and choose points where result was good and re-execute to this
point and then slowly train until best point is reached and predict with this node embedding.

Other/additional idea: classify roc auc>0.6 as category 1 and increate testing/validating and category 2 > 0.7 where
very slowly learnt until it drops and stop immediately. 
"""

x = 0
torch_geometric.seed.seed_everything(x)