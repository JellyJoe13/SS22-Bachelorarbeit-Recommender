import torch
import numpy as np
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from model_workspace.GNN_gcnconv_testspace import GNN_GCNConv_homogen
from edge_batch import EdgeConvolutionBatcher
from data_gen import data_transform_split


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
                "data_mode": 2,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": True
            }
        }

    def should_execute_esc(
            self,
            model_id: int
    ) -> bool:
        return self.model_settings_dict[model_id]["esc"]

    def is_pytorch(
            self,
            model_id: int
    ) -> bool:
        return self.model_settings_dict[model_id]["is_pytorch"]

    def works_on_cuda(
            self,
            model_id: int
    ) -> bool:
        if self.is_pytorch(model_id):
            return self.model_settings_dict[model_id]["cuda_enabled"]
        else:
            return False

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
        # check if graph undirected or directed
        is_directed = True
        dir_edge_index = EdgeConvolutionBatcher.remove_undirected_duplicate_edge(edge_index)
        if dir_edge_index.size(1) == int(edge_index.size(0) / 2):
            is_directed = False
            edge_index = dir_edge_index
        else:
            del dir_edge_index
        # selection
        index = np.arange(edge_index.size(1))
        np.random.shuffle(index)
        choosing_border = int(percentage * edge_index.size(1))
        train_set = edge_index[:, index[:choosing_border]]
        val_set = edge_index[:, index[choosing_border:]]
        if is_directed:
            train_set = torch.cat([train_set, train_set[[1, 0]]], dim=-1)
            val_set = torch.cat([val_set, val_set[[1, 0]]], dim=-1)
        return train_set, val_set

    def load_model_data(
            self,
            model_id: int,
            do_val_split: bool = False,
            split_mode: int = 0
    ) -> dict(int, EdgeConvolutionBatcher):
        """
        Compute the batchers for the model type (may differ if other model types are to be used, e.g. different
        number of convolution layers.)

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
                data.train_pos_edge_index, data.val_pos_edge_index = self.split_off_val_dataset(
                    data.train_pos_edge_index,
                    percentage=0.01)
                # do it for negative edges:
                data.train_neg_edge_index, data.val_neg_edge_index = self.split_off_val_dataset(
                    data.train_neg_edge_index,
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
