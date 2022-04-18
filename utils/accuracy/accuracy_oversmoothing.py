import torch
import numpy as np


def mean_average_distance(
        graph_representation: torch.Tensor,
        pos_edge_index: torch.Tensor
) -> float:
    """
    Function implementing the MAD (mean average distance) metric that should measure smoothness. I do not guarantee for
    correctness or efficiency of function. (Source: https://arxiv.org/pdf/1909.03211.pdf, description of MAD not very
    precise)

    Parameters
    ----------
    graph_representation : torch.Tensor
        graph representation which is output of encoding phase of model
    pos_edge_index : torch.Tensor
        positive edge index indicating which edges are the learn target

    Returns
    -------
    mean average distance : float
    """
    # get graph representation as numpy array
    prepresentation_matrix = graph_representation.detach().numpy()
    # get dimensions of input matrix
    n, h = prepresentation_matrix.shape
    # initialize distance matrix
    d = np.zeros(shape=(n, n))
    # construct distance matrix
    for i in range(n):
        for j in range(n):
            d[i, j] = h[i, :].dot(h[j, :])/(np.linalg.norm(h[i, :]) * np.linalg.norm(h[i, :]))
    # construct mask matrix alias adjacency matrix
    d_tgt = np.zeros(shape=(n, n))
    for edge in pos_edge_index.T:
        d_tgt[edge[0], edge[1]] = d[edge[0], edge[1]]
    # create d_tgt with axon over the d as the d (to save space)
    d = np.divide(np.sum(d_tgt, axis=0),
                  np.sum(d_tgt > 0, axis=0))
    # sum up d and divide it through the sum of positive entries
    return np.sum(d)/np.sum(d > 0)
