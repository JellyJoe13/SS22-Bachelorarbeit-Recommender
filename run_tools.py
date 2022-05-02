import typing
from tqdm.auto import tqdm

import utils.accuracy.accuarcy_bpr
from model_workspace.GNN_minibatch_homogen_GATConv import GNN_GATConv_homogen
from model_workspace.GNN_minibatch_homogen_GCNConv_one import GNN_GCNConv_homogen_basic
from model_workspace.GNN_minibatch_homogen_LGConv_k import GNN_LGConv_homogen_variable
from model_workspace.GNN_minibatch_homogen_SAGEConv import GNN_SAGEConv_homogen
from utils.accuracy import accuracy_recommender
from utils.data_related import edge_batch
from sklearn.metrics import roc_auc_score
import torch
import torch_geometric.data
import torch.nn.functional as F
from datetime import datetime
from model_workspace.GNN_minibatch_homogen_GCNConv_two import GNN_GCNConv_homogen


def train_model_batch(
        model: typing.Union[GNN_GCNConv_homogen,
                            GNN_GATConv_homogen,
                            GNN_GCNConv_homogen_basic,
                            GNN_LGConv_homogen_variable,
                            GNN_SAGEConv_homogen],
        optimizer,
        data: torch_geometric.data.Data,
        loss_function: typing.Callable[[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor], torch.Tensor] = utils.accuracy.accuarcy_bpr.binary_loss_adapter
) -> typing.Tuple[torch.Tensor, float]:
    """
    Helper function that executes the train step for a batch data_related object.

    Parameters
    ----------
    loss_function: typing.Callable[[torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor], torch.Tensor]
        function with three torch.Tensors and returns a torch.Tensor which represents the loss and .backward() will be
        called from the result.
    model : typing.Union[GNN_GCNConv_homogen,
                         GNN_GATConv_homogen,
                         GNN_GCNConv_homogen_basic,
                         GNN_LGConv_homogen_variable,
                         GNN_SAGEConv_homogen]
        model to train on
    optimizer
        optimizer to optimize with
    data : torch_geometric.data.Data
        data_related used for the train process. shall contain x,y, edge_index and pos_edge_index

    Returns
    -------
    loss : torch.Tensor
        loss of the batch data_related object
    roc_auc : float
        ROC AUC of batch data
    """
    # set the model to train mode
    model.train()
    # set the optimizer
    optimizer.zero_grad()
    # fit and predict the edges using the edge_index
    link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
    # calculate loss
    loss = loss_function(link_logits, data.y, data.edge_index)
    # backward optimize loss
    loss.backward()
    # make a step with the optimizer
    optimizer.step()
    # calculate the roc auc
    roc_auc = roc_auc_score(data.y.detach().numpy(), link_logits.sigmoid().detach().numpy())
    # return the loss
    return loss, roc_auc


@torch.no_grad()
def test_model_batch(
        model: typing.Union[GNN_GCNConv_homogen,
                            GNN_GATConv_homogen,
                            GNN_GCNConv_homogen_basic,
                            GNN_LGConv_homogen_variable,
                            GNN_SAGEConv_homogen],
        data: torch_geometric.data.Data
):
    """
    Helper function that executes a simple evaluation for a batch data_related object and determines the logits of the
    edges to predict.

    Parameters
    ----------
    model : typing.Union[GNN_GCNConv_homogen,
                         GNN_GATConv_homogen,
                         GNN_GCNConv_homogen_basic,
                         GNN_LGConv_homogen_variable,
                         GNN_SAGEConv_homogen]
        model to run the test on
    data : torch_geometric.data.Data
        data_related with contains the edges, pos_edges and node data_related to run on

    Returns
    -------
    link_logits : torch.Tensor
        tensor containing the approximations of the edges
    """
    # put the model in evaluation mode
    model.eval()
    # call fit_predict though it will not be fitted due to the @torch.no_grad() annotation
    link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
    return link_logits


def test_model_advanced(
        model: typing.Union[GNN_GCNConv_homogen,
                            GNN_GATConv_homogen,
                            GNN_GCNConv_homogen_basic,
                            GNN_LGConv_homogen_variable,
                            GNN_SAGEConv_homogen],
        batcher: edge_batch.EdgeBatcher,
        model_id: int,
        device,
        epoch: int = 0,
        split_mode: int = 0
) -> typing.Union[typing.Tuple[float, float],
                  typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
    """
    Execute full test creating a roc curve diagram and calculating the accuracy scores.

    Parameters
    ----------
    model : typing.Union[GNN_GCNConv_homogen,
                         GNN_GATConv_homogen,
                         GNN_GCNConv_homogen_basic,
                         GNN_LGConv_homogen_variable,
                         GNN_SAGEConv_homogen]
        model on which to test
    batcher : edge_batch.EdgeConvolutionBatcher
        batcher from which to fetch the batch data_related objects from
    epoch : int
        Additional information that is used for naming the saved plot of the roc curve.
    split_mode : int
        gives information on what split mode has been used. used for determining the name/path of the diagram.

    Returns
    -------
    precision, recall : typing.Union[typing.Tuple[float, float],
    typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]
        precision and recall score of the whole test procedure spanning all batches
    """
    logits_collector = []
    for i in tqdm(range(len(batcher))):
        # load subdata object
        current_batch = batcher(i).to(device)
        # execute test model batch which gets the logits of the edges
        link_logits = test_model_batch(model, current_batch)
        # append labels and logits to storage list
        logits_collector.append(link_logits)
        # detach current batch
        current_batch.detach()
    # create name for roc curve plot
    file_name = "Split-" + str(split_mode) + "_" + model.get_name() + "-" + str(model_id) + "_epoch-" + str(epoch) \
                + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # fuze logits
    logits_collector = torch.cat(logits_collector)
    # create roc plot
    accuracy_recommender.calc_ROC_curve(batcher.target,
                                        logits_collector,
                                        save_as_file=True,
                                        output_file_name=file_name)
    # calculate accuracy
    precision, recall = accuracy_recommender.accuracy_precision_recall(batcher.edges,
                                                                       batcher.target,
                                                                       logits_collector,
                                                                       "both")
    return precision, recall


def test_model_basic(
        model: typing.Union[GNN_GCNConv_homogen,
                            GNN_GATConv_homogen,
                            GNN_GCNConv_homogen_basic,
                            GNN_LGConv_homogen_variable,
                            GNN_SAGEConv_homogen],
        batcher: edge_batch.EdgeConvolutionBatcher,
        device,
        loss_function: typing.Callable[[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor], torch.Tensor] = utils.accuracy.accuarcy_bpr.binary_loss_adapter
):
    """
    Takes model and batcher executes all batches and accumulates the logits and labels to calulate and return the roc
    curve.

    Parameters
    ----------
    loss_function : typing.Callable[[torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor], torch.Tensor]
        loss function which is used to calculate the loss
    model : typing.Union[GNN_GCNConv_homogen,
                         GNN_GATConv_homogen,
                         GNN_GCNConv_homogen_basic,
                         GNN_LGConv_homogen_variable,
                         GNN_SAGEConv_homogen]
        model of the graph neural network
    batcher : edge_batch.EdgeBatcher
        batcher that provides batch data_related objects to process
    device : torch.Device
        device to run the algorithm on

    Returns
    -------
    roc auc score of the tested edges
    """
    # storage for logits
    logits_collector = []
    # iterate over batch objects
    for i in tqdm(range(len(batcher))):
        # get the batch data object
        current_batch = batcher(i).to(device)
        # get logits
        logits = test_model_batch(model, current_batch)
        # put logits in collector
        logits_collector.append(logits)
        # detach data
        current_batch.detach()
    # fuze logits
    logits_collector = torch.cat(logits_collector)
    # calculate roc auc score
    roc_auc = roc_auc_score(batcher.target.cpu(), logits_collector.sigmoid().cpu())
    # calculate loss
    loss = loss_function(logits_collector, batcher.target, batcher.edges)
    # return roc
    return float(loss.detach()), roc_auc


def train_model(
        model: typing.Union[GNN_GCNConv_homogen,
                            GNN_GATConv_homogen,
                            GNN_GCNConv_homogen_basic,
                            GNN_LGConv_homogen_variable,
                            GNN_SAGEConv_homogen],
        batch_list: edge_batch.EdgeBatcher,
        optimizer,
        device,
        loss_function: typing.Callable[[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor], torch.Tensor] = utils.accuracy.accuarcy_bpr.binary_loss_adapter
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[torch.Tensor]]]:
    """
    Execute the training for one epoch. Returns the averaged loss.

    Parameters
    ----------
    loss_function: typing.Callable[[torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor], torch.Tensor]
        function with three torch.Tensors and returns a torch.Tensor which represents the loss and .backward() will be
        called from the result.
    device : torch.Device
        device to run the algorithm on
    model : typing.Union[GNN_GCNConv_homogen,
                         GNN_GATConv_homogen,
                         GNN_GCNConv_homogen_basic,
                         GNN_LGConv_homogen_variable,
                         GNN_SAGEConv_homogen]
        model to train
    batch_list : edge_batch.EdgeBatcher
        batcher from which to fetch the batch data_related objects
    optimizer
        Optimizer to use for training the model

    Returns
    -------
    train_recording : typing.Dict[str, typing.Union[typing.List[float], typing.List[torch.Tensor]]]
        recording dict including loss and roc auc of batches
    """
    # dictionary for recording loss and roc auc
    recording_dict = {
        "loss": [],
        "roc_auc": []
    }
    # for all batch data_related objects stored in batcher do
    for i in tqdm(range(len(batch_list))):
        # fetch batch data object
        current_batch = batch_list(i).to(device)
        # calculate loss and add it to total loss
        current_loss, current_roc_auc = train_model_batch(model,
                                                          optimizer,
                                                          current_batch,
                                                          loss_function).detach() * current_batch.edge_index.size(1)
        # add current results to tracker
        recording_dict["loss"].append(current_loss.detach())
        recording_dict["roc_auc"].append(current_roc_auc)
        # detach batch data object from device
        current_batch.detach()
    # calculate the average loss and return it
    return recording_dict
