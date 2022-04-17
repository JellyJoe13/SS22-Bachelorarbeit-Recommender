import sklearn.metrics
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing


def calc_ROC_curve(
        link_labels: torch.Tensor,
        link_logits: torch.Tensor,
        save_as_file: bool = False,
        output_file_name: str = None
) -> None:
    """
    Function calculating the true positive and false positive rate to compute the ROC curve and plot it via
    matplotlib.

    Parameters
    ----------
    save_as_file : bool
        controls if the plot should be saved.
    output_file_name : str
        Determines the path and file name where the plot SVG should be put (only if save_as_file is set to True).
        NO FILE ENDING LIKE .PNG ETC!
    link_labels : torch.Tensor
        Inputs the true label of the edge - either 0 (inactive) or 1 (active) for each edge to predict
    link_logits : torch.Tensor
        Input the estimation of the edge, which will  be subject to sigmoid interpretion to compute the probability of
        the edge being present in the graph

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
    if save_as_file:
        plt.savefig("plots/" + output_file_name + ".svg")
    else:
        plt.show()
    return


def accuracy_precision_recall(
        edge_index: torch.Tensor,
        link_labels: torch.Tensor,
        link_logits: torch.tensor,
        mode: str = "constant"
) -> typing.Tuple[float, float]:
    """
    Computes the precision and recall of the top k entries. k is determined to be 1% of the input edges. Assumes a
    correct input meaning no edges between types or else they could end up in the accuracy scores.

    Parameters
    ----------
    mode : str
        either "constant" or "relative". Controls if the k for the top k accuracy scores should be constantly set to 100
        or if 1% of the input data_related should be used.
    edge_index : torch.Tensor
        tensor containing the link information which nodes were connected by the labels
    link_labels : torch.Tensor
        tensor containing the true labels of the edges
    link_logits : torch.Tensor
        tensor containing the predicted labels of the edges, probability will be calculated afterwards

    Returns
    -------
    tuple(float, float)
        A tuple containing the precision and recall of the top k entries.
    """
    # assertion section
    assert (mode == "constant") or (mode == "relative")
    # determine the k for the top k entries
    k = int(link_labels.size(0) / 100)
    # set threshold
    threshold = 0.5
    # FASTER VERSION USING PANDAS FRAME
    ids = np.min(edge_index.detach().numpy().T, axis=1)
    # create pandas frame (used for grouping
    df = pd.DataFrame(np.c_[ids,
                            link_labels.detach().numpy(),
                            link_logits.sigmoid().detach().numpy()],
                      columns=['id', 'true_label', 'pred_label'])
    '''
    The following code is inspired/partially copied from the surpriselib documentation and applied for 
    pytorch-geometric. (URL: https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-compute-precision-k-and-recall-k)
    '''
    # create dicts for the id wise precision and recall to be put into
    precisions = {}
    recalls = {}
    # group by id(=user) and calculate the precision and recall for each id grouping
    for df_id, content in df.groupby(by=['id']):
        # sort the data_related according to prediction_label with large entries first
        c = content.sort_values(by=['pred_label'], ascending=False)
        # calculate total number of relations that are active
        n_rel = (c.true_label >= threshold).sum()
        # calculate total number of predictions (first k) that are estimated as active
        n_rec_k = (c.pred_label[:k] >= threshold).sum()
        # calculate true positive k entries
        n_rel_and_rec_k = ((c.true_label[:k] >= threshold) & (c.pred_label[:k] >= threshold)).sum()
        # calculate precision of the id and put it into dict
        precisions[df_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        # calculate recall of the id and put it into dict
        recalls[df_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    # calculate average precisions and recall and return them
    return sum(p for p in precisions.values()) / len(precisions), sum(r for r in recalls.values()) / len(recalls)
