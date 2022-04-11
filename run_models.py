# todo: write whole execution scheme of epochs.... centralized for all models - partly finished
# todo: add validation data - partly finished
# todo: early stopping - partly finished
# todo: measure overfitting with steffen things

# import section
from typing import Union
import numpy as np
import torch

import torch_geometric.data

import edge_batch
import run_tools
from model_workspace.GNN_gcnconv_testspace import GNN_GCNConv_homogen
from data_gen import data_transform_split
from edge_batch import EdgeConvolutionBatcher


class ModelLoader:
    def __init__(
            self
    ):
        self.model_storage = {
            0: GNN_GCNConv_homogen
        }
        self.model_settings_dict = {
            0: {
                "model": 0,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 2
            }
        }

    def load_model(
            self,
            model_id: int
    ):
        """
        Loads the model specified with the model_id and creates the object with the predefined settings.

        Parameters
        ----------
        model_id : int
            defines which model to load, can be interpreted as an index.

        Returns
        -------
        model
            Model to use for recommender/prediction
        """
        if model_id in [0, 1, 2]:
            dict_entry = self.model_settings_dict[model_id]
            return self.model_storage[dict_entry["model"]](num_features_input=dict_entry["num_features_input"],
                                                           num_features_hidden=dict_entry["num_features_hidden"],
                                                           num_features_out=dict_entry["num_features_out"])
        else:
            return None

    @staticmethod
    def split_off_val_dataset(
            edge_index: torch.Tensor,
            percentage: float
    ):
        # todo: is it a problem for double way edges??? maybe add split in edge_batcher?
        index = np.arange(edge_index.size(1))
        np.random.shuffle(index)
        choosing_border = int(percentage*edge_index.size(1))
        train_set = edge_index[:, index[:choosing_border]]
        val_set = edge_index[:, index[choosing_border:]]
        return train_set, val_set

    def load_model_data(
            self,
            model_id: int,
            do_val_split: bool = False,
            split_mode: int = 0
    ) -> dict(int, EdgeConvolutionBatcher):
        """
        Comput the batchers for the model type (may differ if other model types are to be used, e.g. different number of
        convolution layers.)

        Parameters
        ----------
        model_id : int
            defines which model to use and create batches for loaded data
        do_val_split : bool
            defines if a val set shall be created through splitting a part off the trainset
        split_mode : int
            defines which split mode to use, read documentation of data_gen.py in function data_transforma_split to
            learn more

        Returns
        -------
        dict(int, EdgeConvolutionBatcher)
            dict containing the batcher of train, test and val
        """
        # differ between models/model types
        if model_id in [0, 1, 2]:
            # load the data from storage and with datamode and splitmode
            data, id_breakpoint = data_transform_split(data_mode=self.model_settings_dict[model_id]["data_mode"],
                                                       split_mode=split_mode)
            # processing of train data if we should generate a val dataset
            if do_val_split:
                # take 1% of edges from trainset and transfer it to valset. Do for pos and neg edges likewise
                # do it for positive edges:
                data.train_pos_edge_index, data.val_pos_edge_index = self.split_off_val_dataset(data.train_pos_edge_index,
                                                                                                percentage=0.01)
                # do it for negative edges:
                data.train_neg_edge_index, data.val_neg_edge_index = self.split_off_val_dataset(data.train_neg_edge_index,
                                                                                                percentage=0.01)
            # create the batcher for the test edges
            test_batcher = EdgeConvolutionBatcher(data, edge_sample_count=100,
                                                  convolution_depth=2,
                                                  convolution_neighbor_count=100,
                                                  is_directed=False,
                                                  train_test_identifier="test")
            # create a batcher for the train edges
            train_batcher = EdgeConvolutionBatcher(data, edge_sample_count=100,
                                                   convolution_depth=2,
                                                   convolution_neighbor_count=100,
                                                   is_directed=False,
                                                   train_test_identifier="train")
            # create a batcher for val edges if it is desired
            val_batcher = None
            if do_val_split:
                val_batcher = EdgeConvolutionBatcher(data, edge_sample_count=100,
                                                     convolution_depth=2,
                                                     convolution_neighbor_count=100,
                                                     is_directed=False,
                                                     train_test_identifier="val")
            # return the three batchers using a dict to not confuse the batchers with each other.
            return {
                "train": train_batcher,
                "test": test_batcher,
                "val": val_batcher
            }
        else:
            # other model types (maybe full-batch?)
            return None


class EarlyStoppingControl:
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
        # if we do not have 3 entries yet do not stop the training process in any case
        if self.roc_tracker < 3:
            return False
        # get the index of the last entered roc auc values
        last_record_position = len(self.roc_tracker) - 1
        # compute the average of the 2 previous entries before the last one
        comparison_roc_auc = (self.roc_tracker[last_record_position-2] + self.roc_tracker[last_record_position-1]) / 2
        # return true if the last result was lower than the average of the previous two
        return self.roc_tracker[last_record_position] < comparison_roc_auc


def run_epoch(
        model: Union[GNN_GCNConv_homogen],
        optimizer: torch.optim.Optimizer,
        train_batcher: edge_batch.EdgeConvolutionBatcher,
        test_batcher: edge_batch.EdgeConvolutionBatcher,
        model_id: int,
        device
):
    if model_id in [0, 1, 2]:
        loss = run_tools.train_model(model, train_batcher, optimizer)
        roc_auc = run_tools.test_model_basic(model, test_batcher, device)
        return loss, roc_auc
    else:
        return None


def full_test_run(
        model: Union[GNN_GCNConv_homogen],
        batcher: edge_batch.EdgeConvolutionBatcher,
        model_id: int,
        epoch: int,
        split_mode: int,
        device
):
    if model_id in [0, 1, 2]:
        precision, recall = run_tools.test_model_advanced(model, batcher, model_id, device, epoch=epoch,
                                                          split_mode=split_mode)
        return precision, recall
    else:
        return None


def full_experimental_run(
        max_epochs: int,
        model_loader: ModelLoader,
        model_id: int,
        split_mode: int = 0
):
    # load model to test
    model = model_loader.load_model(model_id)
    # load data
    batchers = model_loader.load_model_data(model_id, do_val_split=True, split_mode=split_mode)
    # initialize data and optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    # todo: control device passing
    # run epochs
    for epoch in range(max_epochs):
        loss, roc_auc = run_epoch(model, optimizer, batchers["train"], batchers["test"], model_id, device)
        # todo: add test if val step should be executed
        # todo: early stop thingy
        val_roc_auc = run_tools.test_model_basic(model, batchers["val"], device)
        if epoch % 5 == 0:
            precision, recall = full_test_run(model, batchers["train"], model_id, epoch, split_mode, device)
        # todo: save and or plot data generated
