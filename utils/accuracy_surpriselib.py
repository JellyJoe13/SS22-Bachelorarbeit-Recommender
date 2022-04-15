import surprise.prediction_algorithms.predictions
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import typing


def surpriselib_prediction_to_sklearn(
        predictions: typing.List[surprise.prediction_algorithms.predictions.Prediction]
) -> typing.Tuple[np.ndarray, np.ndarray]:
    # determine how many predictions we have and large the reserved arrays should be
    prediction_count = len(predictions)
    # initialize output parameters y_true and y_score
    y_true = np.zeros(shape=(prediction_count,))
    y_score = np.zeros(shape=(prediction_count,))
    # iterate over predictions and transform it into preallocated arrays
    for index, pred in enumerate(predictions):
        y_true[index] = int(pred.r_ui)
        y_score[index] = pred.est
    return y_true, y_score


def calc_ROC_curve(
        predictions: typing.List[surprise.prediction_algorithms.predictions.Prediction],
        save_as_file: bool = False,
        output_file_name: str = None
) -> None:
    # transform data
    y_true, y_score = surpriselib_prediction_to_sklearn(predictions)
    # use sklearn to get fpr and tpr
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    # calculate the roc_auc (area under the curve)
    roc_auc = metrics.auc(fpr, tpr)
    # PLOT THE ROC CURVE
    '''
    Following code inspired/partly copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label="ROC curve (ROC AUC=%0.2f)" % roc_auc)
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
        predictions: typing.List[surprise.prediction_algorithms.predictions.Prediction],
        mode: str = "constant"
) -> typing.Tuple[float, float]:
    """

    Parameters
    ----------
    predictions : list(surprise.prediction_algorithms.predictions.Prediction),
        contains the predictions of surpriselib acquired with the .test() method. Contains the data with which the
        scores are to be generated.
    mode : str
        either "constant" or "relative". Controls if the k for the top k accuracy scores should be constantly set to 100
        or if 1% of the input data should be used.

    Returns
    -------
    precision, recall : tuple(float, float)
        precision and recall top k accuracy scores computed with the input data
    """
    # assertion section
    assert (mode == "constant") or (mode == "relative")
    # transform predictions to numpy arrays
    y_true = np.array([pred.r_ui for pred in predictions])
    y_score = np.array([pred.est for pred in predictions])
    ids = np.array([pred.uid for pred in predictions])
    # determine k for the top k entries
    if mode == "constant":
        k = 100
    else:
        k = int(y_true.shape[0] / 100)
    # define threshold
    threshold = 0.5
    # create pandas frame
    df = pd.DataFrame(np.c_[ids, y_true, y_score],
                      columns=['id', 'true_label', 'pred_label'])
    # create dicts for the id wise precision and recall to be put into
    precisions = {}
    recalls = {}
    # group by id(=user) and calculate the precision and recall for each id grouping
    for df_id, content in df.groupby(by=['id']):
        # sort the data according to prediction_label with large entries first
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
