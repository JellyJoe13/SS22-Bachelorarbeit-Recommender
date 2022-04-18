class EarlyStoppingControl:
    """
    Class used for controlling if the training process should stop before the max iteration count is reached.
    """
    def __init__(self):
        self.roc_tracker = []

    def put_in_roc_auc(
            self,
            roc_auc: float
    ) -> None:
        """
        Put new roc auc value into the EarlyStoppingControl class.

        Parameters
        ----------
        roc_auc : float
            last measured roc auc value

        Returns
        -------
        Nothing
        """
        # save the supplied roc auc into list
        self.roc_tracker.append(roc_auc)
        return

    def get_should_stop(
            self
    ) -> bool:
        """
        Determines if the learning should stop or not.

        Returns
        -------
        bool
            True if the learning should stop, False if it could continue
        """
        # todo: rework this as we may need a condition to do at least 10 iterations
        # if we do not have 3 entries yet do not stop the training process in any case
        if self.roc_tracker < 3:
            return False
        # get the index of the last entered roc auc values
        last_record_position = len(self.roc_tracker) - 1
        # compute the average of the 2 previous entries before the last one
        comparison_roc_auc = (self.roc_tracker[last_record_position - 2] + self.roc_tracker[
            last_record_position - 1]) / 2
        # return true if the last result was lower than the average of the previous two
        return self.roc_tracker[last_record_position] < comparison_roc_auc
