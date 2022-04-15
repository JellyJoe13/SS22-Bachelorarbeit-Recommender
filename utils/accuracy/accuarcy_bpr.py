import torch
from torch_geometric.nn import functional as F
"""
This whole file inspired by the Standford CS224W Machine Learning with Graphs course project and utilized the loss
function from it to be used in this project. Read more about the project via
https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377 or find the code in the notepad
available via 
https://colab.research.google.com/drive/1VQTBxJuty7aLMepjEYE-d7E9kjo51CA1?usp=sharing#scrollTo=bwrPmvXPow5q.
General information about the project and its contributors can be acquired here: http://web.stanford.edu/class/cs224w/
Important note: Original content is heavily modified to fit the usage in this project.
"""


def compute_bpr_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
):
    # create every combination of every pos score with every neg score
    combinations = torch.cat([pos_entry-neg_scores for pos_entry in pos_scores])
    # calculate the mean of these combinations
    bpr_loss = torch.mean(F.softplus(combinations))

    return bpr_loss
