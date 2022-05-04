import typing

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from sklearn.metrics import roc_auc_score
# adding parent folder to execution path
# this part fixing the imports from the parent directory are copied from
# https://www.geeksforgeeks.org/python-import-from-parent-directory/
import sys
import os
from datetime import datetime

import utils.accuracy.accuarcy_bpr

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# import section
from utils.accuracy.accuracy_recommender import accuracy_precision_recall, calc_ROC_curve


class GNN_homogen_chemData_GCN(torch.nn.Module):
    """
    Basic implementation of GNN, running in full batch mode.

    Inspired by content shown in the pytorch geomtric tutorial
    (https://antoniolonga.github.io/Pytorch_geometric_tutorials/posts/post12.html) from Antionio Longa
    """

    def __init__(
            self,
            num_features_input: int,
            num_features_hidden: int,
            num_features_out: int = 64
    ):
        """
        Initializing function of GNN_homogen_chemData_GCN.

        Parameters
        ----------
        num_features_input : int
            Specifies how many input features to expect and work with in the input data_related
        num_features_hidden : int
            Specifies how many hidden nodes of convolution layer there should be in the model
        num_features_out : int
            Specifies how many output nodes there are - how many dimensions the node embedding should have
        """
        super(GNN_homogen_chemData_GCN, self).__init__()
        self.conv1 = GCNConv(num_features_input, num_features_hidden)
        self.conv2 = GCNConv(num_features_hidden, num_features_out)

    def encode(
            self,
            data: torch_geometric.data.Data
    ) -> torch.Tensor:
        """
        Function for encoding the data_related - generating the node embedding.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Supplies the data_related object with the node features and known positive edges (as train_pos_edge_index)

        Returns
        -------
        torch.Tensor
            Tensor containing the node embeddings that were computed using the neural network structure
        """
        # first convolution layer and input sent through relu to remove possible negative edges
        x = self.conv1(data.x.relu(), data.train_pos_edge_index)
        # relu to remove intermediary negative edges from convolution layer output
        x = x.relu()
        # second convolution layer
        x = self.conv2(x, data.train_pos_edge_index)
        # return the node embeddings
        return x

    def decode(
            self,
            node_embeddings: torch.Tensor,
            pos_edge_index: torch.Tensor,
            neg_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Function for using the node embeddings generated in the function encode() and using it with the edge information
        to compute the approximation of the edge.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Tensor containing the node embeddings generated by the function encode()
        pos_edge_index : torch.Tensor
            Tensor containing the edge information of the positive edges
        neg_edge_index : torch.Tensor
            Tensor containing the edge information of the negative edges

        Returns
        -------
        logits : torch.Tensor
            Tensor contining logits which are approximations of the edge - converted to probabilities using sigmoid()
            function.
        """
        # concatenate edges to get all the edges to decode
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # calculate the edge prediction using the columns of the node_embeddings
        logits = (node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]).sum(dim=-1)
        # return the approximations - also called logits
        return logits

    @staticmethod
    def get_link_labels(
            pos_edge_index: torch.Tensor,
            neg_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Creates and outputs a tensor with 0 and 1 entries corresponding to the positiveness or negativeness of an edge.

        Output tensor has length pos_edge_index.size(1)+neg_edge_index.size(1) and a corresponding amount of 1s
        (representing the positive edges) and 0s (representing the negative edges).

        Parameters
        ----------
        device : torch.device
            device on which the tensor shall be deployed. Learn more about this in the pytorch documentation
        pos_edge_index : torch.Tensor
            positive data_related edges
        neg_edge_index : torch.Tensor
            negative data_related edges

        Returns
        -------
        torch.Tensor
            Tensor with 0 and 1 entries corresponding to the positiveness or negativeness of an edge.
        """
        # calculates the length of the output tensor
        output_tensor_length = pos_edge_index.size(1) + neg_edge_index.size(1)
        # creates the output tensor with zeros
        link_labels = torch.zeros(output_tensor_length, dtype=torch.float)
        # change the labels corresponding to positive edges to ones
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    @staticmethod
    def get_name() -> str:
        """
        Defines a name of the model that is used if an output file is generated.

        Returns
        -------
        str
            Name of the model, used in output files
        """
        return "GNN_GCNConv_homogen_fullbatch"


def train(
        model: GNN_homogen_chemData_GCN,
        optimizer: torch.optim.Optimizer,
        data: torch_geometric.data.Data,
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.binary_cross_entropy_with_logits
):
    """
    Function used to execute one training epoch with the objects supplied.

    Parameters
    ----------
    loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        function with two torch.Tensors and returns a torch.Tensor which represents the loss and .backward() will be
        called from the result.
    model : GNN_homogen_chemData_GCN
        model which will be trained
    optimizer : torch.optim.Optimizer
        optimizer used for training process
    data : torch.data_related.Data
        data_related on which the model will be trained

    Returns
    -------
    loss
        Loss of the training epoch
    """
    model.train()
    optimizer.zero_grad()
    link_logits = model.decode(model.encode(), data.train_pos_edge_index, data.train_neg_edge_index)
    link_labels = model.get_link_labels(data.train_pos_edge_index, data.train_neg_edge_index)
    loss = loss_function(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


def train_with_roc_auc(
        model: GNN_homogen_chemData_GCN,
        optimizer: torch.optim.Optimizer,
        data: torch_geometric.data.Data,
        loss_function: typing.Callable[[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor], torch.Tensor] = utils.accuracy.accuarcy_bpr.binary_loss_adapter
) -> typing.Tuple[torch.Tensor, float]:
    """
    Function which performs a training step for a data object on the model GNN_homogen_chemData_GCN and returns the loss
    and roc auc of the data being trained.

    Parameters
    ----------
    model : GNN_homogen_chemData_GCN
        model to execute the train operation on.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    data : torch.data.Data
        data to use for training the model
    loss_function : typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        loss function to use for training to determine the error

    Returns
    -------
    loss : torch.Tensor
        loss for the data trained
    roc_auc : float
        roc auc for the data trained on the model
    """
    # execute training
    model.train()
    optimizer.zero_grad()
    link_logits = model.decode(model.encode(data), data.train_pos_edge_index, data.train_neg_edge_index)
    link_labels = model.get_link_labels(data.train_pos_edge_index, data.train_neg_edge_index)
    loss = loss_function(
        link_logits,
        link_labels,
        torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim=-1)
    )
    loss.backward()
    optimizer.step()
    # calculate roc auc
    roc_auc = roc_auc_score(link_labels.cpu(), link_logits.sigmoid().cpu())
    return loss.detach(), roc_auc


@torch.no_grad()
def test(
        model: GNN_homogen_chemData_GCN,
        data: torch_geometric.data.Data,
        learn_model: str = "test"
):
    """
    Function used for executing a test of the model using the data_related stored in the data_related object.

    Parameters
    ----------
    model : GNN_homogen_chemData_GCN
        model used for testing
    data : torch_geometric.data.Data
        data_related which contains the test data_related for the test process
    learn_model : str
        learn model which defines which part of the dataset should be tasted. Options: "test", "train", "val". Edges for
        this learn set must exist in data_related object

    Returns
    -------
    roc_auc_score
        Value containing the roc auc score which represents the area under the roc curve. See documentation of scikit
        learn to learn more about this score.
    """
    model.eval()
    link_probs = model.decode(model.encode(data),
                              data[learn_model + "_pos_edge_index"],
                              data[learn_model + "_neg_edge_index"]).sigmoid()
    link_labels = model.get_link_labels(data[learn_model + "_pos_edge_index"],
                                        data[learn_model + "_neg_edge_index"])
    return roc_auc_score(link_labels.cpu(), link_probs.cpu())


@torch.no_grad()
def test_with_loss(
        model: GNN_homogen_chemData_GCN,
        data: torch_geometric.data.Data,
        learn_model: str = "test",
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.binary_cross_entropy_with_logits
):
    """
    Function used for executing a test of the model using the data_related stored in the data_related object.

    Parameters
    ----------
    model : GNN_homogen_chemData_GCN
        model used for testing
    data : torch_geometric.data.Data
        data_related which contains the test data_related for the test process
    learn_model : str
        learn model which defines which part of the dataset should be tasted. Options: "test", "train", "val". Edges for
        this learn set must exist in data_related object
    loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.binary_cross_entropy_with_logits
        function with two torch.Tensors and returns a torch.Tensor which represents the loss.

    Returns
    -------
    roc_auc_score
        Value containing the roc auc score which represents the area under the roc curve. See documentation of scikit
        learn to learn more about this score.
    """
    model.eval()
    link_logits = model.decode(model.encode(data),
                               data[learn_model + "_pos_edge_index"],
                               data[learn_model + "_neg_edge_index"])
    link_labels = model.get_link_labels(data[learn_model + "_pos_edge_index"],
                                        data[learn_model + "_neg_edge_index"])
    loss = loss_function(
        link_logits,
        link_labels,
        torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim=-1)
    )
    roc_auc = roc_auc_score(link_labels.cpu(), link_logits.sigmoid().cpu())
    return float(loss.detach()), roc_auc


@torch.no_grad()
def full_test(
        model: GNN_homogen_chemData_GCN,
        data: torch_geometric.data.Data,
        model_id: int,
        epoch: int = 0,
        split_mode: int = 0
):
    """
    Executes a full test of the model computing the precision, recall and roc curve and plot the latter diagram.

    Parameters
    ----------
    split_mode : int
        Supplies split mode id for filename choosing
    epoch : int
         Supplies epoch for filename choosing
    model_id : int
        Supplies model id for filename choosing
    model : GNN_homogen_chemData_GCN
        Model which will be subject to the full_test testing routine
    data : torch_geometric.data.Data
        Data which contains the test data_related

    Returns
    -------
    Nothing, but plots roc curve and prints accuracy values
    """
    model.eval()
    link_logits = model.decode(model.encode(), data.test_pos_edge_index, data.test_neg_edge_index)
    print(link_logits)
    link_labels = model.get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)
    # compute recall and precision
    precision, recall = accuracy_precision_recall(
        torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1),
        link_labels,
        link_logits,
        mode="both"
    )
    print("precision:", precision, "\nrecall:", recall)
    # create file name for plot
    file_name = "Split-" + str(split_mode) + "/" + model.get_name() + "-" + str(model_id) + "_epoch-" + str(epoch) \
                + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # plot ROC CURVE
    calc_ROC_curve(link_labels,
                   link_logits,
                   save_as_file=True,
                   output_file_name=file_name)
    return precision, recall
