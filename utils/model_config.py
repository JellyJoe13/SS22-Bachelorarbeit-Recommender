import surprise
import torch
import numpy as np
import sys
import os
import typing

import torch_geometric.data

import utils.accuracy.accuarcy_bpr

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from model_workspace.GNN_minibatch_homogen_GCNConv_two import GNN_GCNConv_homogen
from edge_batch import EdgeConvolutionBatcher
from data_gen import data_transform_split
from model_workspace.GNN_fullbatch_homogen_GCNConv import GNN_homogen_chemData_GCN
from surprise import SVD
from torch.nn import functional as F


# todo: extend to surpriselib
class ModelLoader:
    def __init__(
            self
    ):
        self.model_storage = {
            -1: SVD,
            0: GNN_homogen_chemData_GCN,
            1: GNN_GCNConv_homogen,
        }
        self.loss_function_storage = {
            "binary": F.binary_cross_entropy_with_logits,
            "bpr": utils.accuracy.accuarcy_bpr.adapter_brp_loss_GNN
        }
        self.model_settings_dict = {
            # surpriselib model with baseline recommender
            -1: {
                "model": -1,
                "data_mode": 0,
                "esc": False,
                "is_pytorch": False,
                "is_batching": False
            },
            # ONEDIGIT SECTION - FULLBATCH
            # - DATA SECTION
            #   + pytorch homogen fullbatch GCNConv-0 binaryloss
            0: {
                "model": 0,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 2,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": False,
                "is_batched": False,
                "loss": "binary"
            },
            #   + pytorch homogen fullbatch GCNConv-0 bprloss
            1: {
                "model": 0,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 2,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": False,
                "is_batched": False,
                "loss": "bpr"
            },
            # - NODATA SECTION
            2: {
                "model": 0,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 1,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": False,
                "is_batched": False,
                "loss": "binary"
            },
            # TEN SECTION - MINIBATCH-DATA
            # - pytorch homogen minibatch GCNConv-0 binaryloss
            10: {
                "model": 1,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 2,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": True,
                "is_batched": True,
                "loss": "binary"
            },
            # - pytorch homogen minibatch GCNConv-0 bprloss
            11: {
                "model": 1,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 2,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": True,
                "is_batched": True,
                "loss": "bpr"
            },
            # TWENTY SECTION - MINIBATCH-NODATA
            # - pytorch homogen minibatch GCNConv-0 binaryloss
            10: {
                "model": 1,
                "num_features_input": 205,
                "num_features_output": 64,
                "num_features_hidden": 100,
                "data_mode": 1,
                "esc": True,
                "is_pytorch": True,
                "cuda_enabled": True,
                "is_batched": True,
                "loss": "binary"
            },
        }

    def should_execute_esc(
            self,
            model_id: int
    ) -> bool:
        return self.model_settings_dict[model_id]["esc"]

    def get_loss_function(
            self,
            model_id: int
    ) -> typing.Tuple[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        loss_name = self.model_settings_dict[model_id]["loss"]
        return loss_name, self.loss_function_storage[loss_name]

    def is_pytorch(
            self,
            model_id: int
    ) -> bool:
        return self.model_settings_dict[model_id]["is_pytorch"]

    def is_batched(
            self,
            model_id: int
    ) -> bool:
        if self.is_pytorch(model_id):
            return self.model_settings_dict[model_id]["is_batched"]
        else:
            return False

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
        if self.is_pytorch(model_id):
            # load corresponding dict entry
            dict_entry = self.model_settings_dict[model_id]
            # it is a pytorch model. determine which one
            model = self.model_storage[dict_entry["model"]]
            if model == GNN_homogen_chemData_GCN:
                return model(num_features_input=dict_entry["num_features_input"],
                             num_features_hidden=dict_entry["num_features_hidden"],
                             num_features_out=dict_entry["num_features_out"])
            elif model == GNN_GCNConv_homogen:
                return model(num_features_input=dict_entry["num_features_input"],
                             num_features_hidden=dict_entry["num_features_hidden"],
                             num_features_out=dict_entry["num_features_out"])
            else:
                return None
        else:
            # model is not pytorch, hence it is surpriselib
            return self.model_storage[self.model_storage[model_id]]()

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
    ) -> typing.Union[typing.Dict[int, EdgeConvolutionBatcher],
                      torch_geometric.data.Data,
                      typing.Tuple[surprise.trainset.Trainset, list]]:
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

        Returns in case of pytorch and minibatching
        -------------------------------------------
        dict(int, EdgeConvolutionBatcher)
            dict containing the batcher of train, test and val

        Returns in case of pytorch and fullbatch mode
        ---------------------------------------------
        torch_geometric.data.Data
            Data object containing the whole dataset

        Return in case of surpriselib
        -----------------------------
        tuple(surprise.trainset.Trainset, list)
            Tuple containing the trainset and testset which is needed for surpriselib
        """
        # todo: double usage of type parameters is_pytorch and data mode
        # differ if model to load is pytorch or not
        if self.is_pytorch(model_id):
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
            # determine if we work with minibatching or fullbatching
            if self.is_batched(model_id):
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
                # fullbatch mode
                return data
        else:
            # surpriselib part
            return data_transform_split(0, split_mode)
