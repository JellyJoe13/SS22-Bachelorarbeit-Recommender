import sklearn.metrics
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd


def calc_ROC_curve(
        link_labels: torch.Tensor,
        link_logits: torch.Tensor,
        save_as_file: bool = False,
        output_path: str = None
) -> None:
    """
    Function calculating the true positive and false positive rate to compute the ROC curve and plot it via
    matplotlib.

    Parameters
    ----------
    save_as_file
        bool : controls if the plot should be saved.
    output_path
        str : Determines the path and file name where the plot SVG should be put (only if save_as_file is set to True)
    link_labels
        torch.Tensor : Inputs the true label of the edge - either 0 (inactive) or 1 (active) for each edge to predict
    link_logits
        torch.Tensor : Input the estimation of the edge, which will  be subject to sigmoid interpretion to compute the
        probability of the edge being present in the graph

    Returns
    -------
    Nothing, but plots the ROC Curve and saves it if the parameters state it
    """
    # calculate the tpr and fpr using the function from sklearn
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(link_labels.detach().numpy(),
                                                                           link_logits.sigmoid().detach().numpy())
    # calculate the ROC AUC (area under curve) to put it into the graph as well
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    # plot the ROC CURVE
    '''
    Following code is inspired or partially copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color="darkorange", label="ROC curve (ROC AUC=%0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="blue", linestyle="--")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC CURVE")
    plt.legend(loc="lower right")
    plt.show()
    return


def accuracy_precision_recall(
        edge_index: torch.Tensor,
        link_labels: torch.Tensor,
        link_logits: torch.tensor,
        id_breakpoint: int
) -> tuple(int, int):
    """
    Computes the precision and recall of the top k entries. k is determined to be 1% of the input edges.

    Parameters
    ----------
    edge_index
        torch.Tensor : tensor containing the link information which nodes were connected by the labels
    link_labels
        torch.Tensor : tensor containing the true labels of the edges
    link_logits
        torch.Tensor : tensor containing the predicted labels of the edges, probability will be calculated afterwards
    id_breakpoint
        int : Number indexing until which index of the GNN-id the original assay id and compound id belong

    Returns
    -------
    tuple(int, int) : Returns a tuple containing the precision and recall of the top k entries.
    """
    # determine the k for the top k entries
    k = int(link_labels.size(0)/100)
    # set threshold
    threshold = 0.5
    df = pd.DataFrame(np.c_[np.transpose(edge_index.detach().numpy()),
                            link_labels.detach().numpy(),
                            link_logits.sigmoid().detach().numpy()],
                      columns=['id1', 'id2', 'true_label', 'pred_label'])
    df['id1'] = df['id1'].map(lambda x: x if x < id_breakpoint else 0)
    df['id2'] = df['id1'].map(lambda x: x if x < id_breakpoint else 0)
    df['id'] = df['id1'].to_numpy()+df['id2'].to_numpy()
    df.drop(columns=['id1', 'id2'])
