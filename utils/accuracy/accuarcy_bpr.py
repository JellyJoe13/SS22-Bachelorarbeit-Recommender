import torch
"""
This whole file inspired by the Standford CS224W Machine Learning with Graphs course project and utilized the loss
function from it to be used in this project. Read more about the project via
https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377 or find the code in the notepad
available via 
https://colab.research.google.com/drive/1VQTBxJuty7aLMepjEYE-d7E9kjo51CA1?usp=sharing#scrollTo=bwrPmvXPow5q.
General information about the project and its contributors can be acquired here: http://web.stanford.edu/class/cs224w/
Important note: Original content is heavily modified to fit the usage in this project.
"""


# todo: needs tremendous reworking as allocating way to much time and memory:
# use log(sigmoid()) instead of softmax
# use sum over all positive/negative scores per user and then apply sigmoid, log and mean over all users in set
# redefine batching for using whole users if possible: sklearn GroupShuffleSplit
def compute_bpr_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
) -> torch.Tensor:
    """
    Function computing the bpr loss given the pos and neg scores.

    Parameters
    ----------
    pos_scores : torch.Tensor
        positive scores meaning the logits not the probabilities
    neg_scores : torch.Tensor
        negative scores meaning the logits not the probabilities

    Returns
    -------
    bpr_loss : torch.Tensor
        Loss that was computed in the function. Used to train model.
    """
    # create every combination of every pos score with every neg score
    combinations = torch.cat([pos_entry-neg_scores for pos_entry in pos_scores])
    # calculate the mean of these combinations
    bpr_loss = torch.mean(torch.nn.functional.softplus(combinations))
    # returns the computed loss
    return bpr_loss


def adapter_brp_loss_GNN(
        link_logits: torch.Tensor,
        link_labels: torch.Tensor
) -> torch.Tensor:
    """
    Function used as a parameter for function compute_bpr_loss() in case the pos and neg scores were not computed yet.

    Parameters
    ----------
    link_logits : Approximations of the Graph neural networks for the edges
    link_labels : True label of the edges

    Returns
    -------
    bpr_loss : torch.Tensor
        bpr loss that was computed given the input tensors
    """
    # separate pos and neg labels
    pos_scores = link_logits[link_labels == 1]
    neg_scores = link_logits[link_labels == 0]
    # call real loss function and return value
    return compute_bpr_loss(pos_scores, neg_scores)
