import surprise.prediction_algorithms.predictions
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
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
    # transform data_related
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
) -> typing.Union[typing.Tuple[float, float],
                  typing.Tuple[typing.Tuple[float, float],
                               typing.Tuple[float, float]]]:
    """

    Parameters
    ----------
    predictions : list(surprise.prediction_algorithms.predictions.Prediction),
        contains the predictions of surpriselib acquired with the .test() method. Contains the data_related with which the
        scores are to be generated.
    mode : str
        either "constant" or "relative". Controls if the k for the top k accuracy scores should be constantly set to 100
        or if 1% of the input data_related should be used.

    Returns
    -------
    tuple(float, float)
        A tuple containing the precision and recall of the top k entries in case the mode is either "constant" or
        "relative"
    tuple(tuple(float, float), tuple(float, float))
        If mode is "both" returns a tuple containing both precision and recall scores for both "constant" and "relative"
        mode in a fashion (precision_constant, precision_relative), (recall_constant, recall_relative)
    """
    # assertion section
    assert (mode == "constant") or (mode == "relative") or (mode == "both")
    # transform predictions to numpy arrays
    y_true = np.array([pred.r_ui for pred in predictions])
    y_score = np.array([pred.est for pred in predictions])
    ids = np.array([pred.uid for pred in predictions])
    # set modes
    mode_constant = mode == "constant"
    mode_relative = mode == "relative"
    # determine the k for the top k entries
    k_r = int(y_true.shape[0] / 100)
    k_c = 100
    # define threshold
    threshold = 0.5
    # create pandas frame
    df = pd.DataFrame(np.c_[ids, y_true, y_score],
                      columns=['id', 'true_label', 'pred_label'])
    # create dicts for the id wise precision and recall to be put into
    if mode_relative:
        precisions_r = {}
        recalls_r = {}
    if mode_constant:
        precisions_c = {}
        recalls_c = {}
    # group by id(=user) and calculate the precision and recall for each id grouping
    for df_id, content in df.groupby(by=['id']):
        # sort the data_related according to prediction_label with large entries first
        c = content.sort_values(by=['pred_label'], ascending=False)
        # calculate total number of relations that are active
        n_rel = (c.true_label >= threshold).sum()
        # calculate total number of predictions (first k) that are estimated as active
        if mode_constant:
            n_rec_k_c = (c.pred_label[:k_c] >= threshold).sum()
        if mode_relative:
            n_rec_k_r = (c.pred_label[:k_r] >= threshold).sum()
        # calculate true positive k entries
        if mode_constant:
            n_rel_and_rec_k_c = ((c.true_label[:k_c] >= threshold) & (c.pred_label[:k_c] >= threshold)).sum()
        if mode_relative:
            n_rel_and_rec_k_r = ((c.true_label[:k_r] >= threshold) & (c.pred_label[:k_r] >= threshold)).sum()
        # calculate precision of the id and put it into dict
        if mode_constant:
            precisions_c[df_id] = n_rel_and_rec_k_c / n_rec_k_c if n_rec_k_c != 0 else 0
        if mode_relative:
            precisions_r[df_id] = n_rel_and_rec_k_r / n_rec_k_r if n_rec_k_r != 0 else 0
        # calculate recall of the id and put it into dict
        if mode_constant:
            recalls_c[df_id] = n_rel_and_rec_k_c / n_rel if n_rel != 0 else 0
        if mode_relative:
            recalls_r[df_id] = n_rel_and_rec_k_r / n_rel if n_rel != 0 else 0
    # calculate average precisions and recall
    if mode_constant:
        mean_prec_c = sum(p for p in precisions_c.values()) / len(precisions_c)
        mean_rec_c = sum(r for r in recalls_c.values()) / len(recalls_c)
    if mode_relative:
        mean_prec_r = sum(p for p in precisions_r.values()) / len(precisions_r)
        mean_rec_r = sum(r for r in recalls_r.values()) / len(recalls_r)
    # return values
    if mode_constant and mode_relative:
        return (mean_prec_c, mean_prec_r), (mean_rec_c, mean_rec_r)
    elif mode_constant:
        return mean_prec_c, mean_rec_c
    else:
        return mean_prec_r, mean_rec_r
