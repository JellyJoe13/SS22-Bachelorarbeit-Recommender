import typing

import run_tools
from model_workspace.GNN_minibatch_homogen_GCNConv_one import GNN_GCNConv_homogen_basic
from utils.accuracy.accuarcy_bpr import bpr_loss_revised
from utils.data_related.data_gen import data_transform_split
from torch_geometric.data import Data
import torch
from utils.data_related.edge_batch import EdgeBatcher
from tqdm.auto import tqdm
import numpy as np


class RecommenderBScUrban:
    """
    Recommender that contains the best model determined in the bachelor thesis.
    """
    def __init__(
            self,
            batch_size: int,
            data_path: str = "df_assay_entries.csv"
    ):
        """
        Initializes the recommender with GCN - 1 layer - bpr loss - chemical descriptors - zero experiment values

        Parameters
        ----------
        batch_size : int
            batch size to use - optimally the largest possible batch size that is able to run on the CUDA system.
        data_path : str
            path to the data used for input data loading
        """
        assert batch_size > 0
        self.batch_size = batch_size
        self.__load_data(data_path, batch_size)
        self.__prepare_recommender_setup()

    def __load_data(self, data_path, batch_size):
        data, self.data_handler = data_transform_split(
            data_mode=2,
            split_mode=0,
            path=data_path
        )
        training_data = Data(
            x=data.x,
            train_pos_edge_index=torch._cat([data.train_pos_edge_index, data.test_pos_edge_index], axis=1),
            train_neg_edge_index=torch._cat([data.train_neg_edge_index, data.test_neg_edge_index], axis=1),
        )
        self.train_data = EdgeBatcher(
            data=training_data,
            num_selection_edges=batch_size,
            mode="train"
        )
        return

    def __prepare_recommender_setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNN_GCNConv_homogen_basic(205, 64).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.loss_function = bpr_loss_revised
        return

    def initialize(self, epoch: int = 25):
        """
        Initializes recommender meaning it executes training epochs.

        Parameters
        ----------
        epoch : int
            Number of epochs to run before predicting is possible

        Returns
        -------
        None
        """
        for i in tqdm(range(epoch)):
            for index in range(len(self.train_data)):
                batch = self.train_data(index).to(self.device)
                loss, _ = run_tools.train_model_batch(
                    model=self.model,
                    optimizer=self.optimizer,
                    data=batch,
                    loss_function=self.loss_function
                )
                batch = batch.detach()
        return

    @torch.no_grad()
    def predict(
            self,
            cids: typing.Union[list, np.ndarray],
            aids: typing.Union[list, np.ndarray]
    ) -> np.ndarray:
        """
        Function to predict molecule-experiment pairs.

        Parameters
        ----------
        cids : typing.Union[list, np.ndarray]
            molecule ids to predict. works pairwise with aid parameter. will not compute each missing value for an input
            cid. (of the total activity matrix)
        aids : typing.Union[list, np.ndarray]
            experiment ids to predict. works pairwise with cid parameter. will not compute each missing value for an
            input aid. (of the total activity matrix)

        Returns
        -------
        predictions : np.ndarray
            numpy array containing the probabilities for the input molecule-experiment pairs to be active.
        """
        # convert lists to numpy arrays
        if isinstance(cids, list):
            cids = np.array(cids)
        if isinstance(aids, list):
            aids = np.array(aids)
        # transform ids to edges
        edges = self.data_handler.create_mapped_edges(cids, aids)
        # create collector for predictions
        prediction_collector = []
        for start_idx in range(0, edges.size(1), self.batch_size):
            # determine range
            end_idx = start_idx + self.batch_size
            if end_idx > edges.size(1):
                end_idx = edges.size(1)
            # get prediction
            prediction = self.model.fit_predict(
                self.train_data.original_data.x.to(self.device),
                edges[:, start_idx:end_idx].to(self.device),
                self.train_data.original_data.train_pos_edge_index.to(self.device)
            ).sigmoid().detach().cpu().numpy()
            # append prediction to collector
            prediction_collector.append(prediction)
        # merge predictions
        return np.concatenate(prediction_collector)
