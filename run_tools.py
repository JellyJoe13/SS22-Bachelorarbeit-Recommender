import typing

import accuracy_recommender
import edge_batch
import sklearn.metrics
import torch
import torch_geometric.data
import torch.nn.functional as F
from datetime import datetime
from model_workspace.GNN_gcnconv_testspace import GNN_GCNConv_homogen


def train_model_batch(
        model: GNN_GCNConv_homogen,
        optimizer,
        data: torch_geometric.data.Data,
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.binary_cross_entropy_with_logits
) -> torch.Tensor:
    """
    Helper function that executes the train step for a batch data object.

    Parameters
    ----------
    loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        function with two torch.Tensors and returns a torch.Tensor which represents the loss and .backward() will be
        called from the result.
    model : GNN_GCNConv_homogen
        model to train on
    optimizer
        optimizer to optimize with
    data : torch_geometric.data.Data
        data used for the train process. shall contain x,y, edge_index and pos_edge_index

    Returns
    -------
    loss : torch.Tensor
        loss of the batch data object
    """
    # set the model to train mode
    model.train()
    # set the optimizer
    optimizer.zero_grad()
    # fit and predict the edges using the edge_index
    link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
    # get true predictions in form of a torch Tensor for loss computation
    link_labels = data.y
    # calculate loss
    loss = loss_function(link_logits, link_labels)
    # backward optimize loss
    loss.backward()
    # make a step with the optimizer
    optimizer.step()
    # return the loss
    return loss


@torch.no_grad()
def test_model_batch(
        model: GNN_GCNConv_homogen,
        data: torch_geometric.data.Data
):
    """
    Helper function that executes a simple evaluation for a batch data object and determines the logits of the edges to
    predict.

    Parameters
    ----------
    model : GNN_GCNConv_homogen
        model to run the test on
    data : torch_geometric.data.Data
        data with contains the edges, pos_edges and node data to run on

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
        model: GNN_GCNConv_homogen,
        batcher: edge_batch.EdgeConvolutionBatcher,
        model_id: int,
        device,
        epoch: int = 0,
        split_mode: int = 0
) -> typing.Tuple[float, float]:
    """
    Execute full test creating a roc curve diagram and calculating the accuracy scores.

    Parameters
    ----------
    model : GNN_GCNConv_homogen
        model on which to test
    batcher : edge_batch.EdgeConvolutionBatcher
        batcher from which to fetch the batch data objects from
    epoch : int
        Additional information that is used for naming the saved plot of the roc curve.
    split_mode : int
        gives information on what split mode has been used. used for determining the name/path of the diagram.

    Returns
    -------
    precision, recall : tuple(float, float)
        precision and recall score of the whole test procedure spanning all batches
    """
    # poll first element from batcher stack
    current_batch, retranslation_dict = batcher.next_element()
    # transfer data to device
    current_batch = current_batch.to(device)
    # create empty lists to append the results to in the while loop
    batch_loop_storage = []
    edge_index_transformed = []
    # for all batch data objects in batcher do
    while current_batch:
        # execute test model batch which gets the logits of the edges
        link_logits = test_model_batch(model, current_batch)
        # append labels and logits to storage list
        batch_loop_storage.append((current_batch.y,
                                   link_logits))
        # poll next element
        current_batch, retranslation_dict = batcher.next_element()
        # write and transform edge information to new edge_index (in list form)
        for edge in current_batch.edge_index.T:
            edge_index_transformed.append([retranslation_dict[edge[0]], retranslation_dict[edge[1]]])
    # fuze batch results
    link_logits = torch.cat([i[0] for i in batch_loop_storage])
    link_labels = torch.cat([i[1] for i in batch_loop_storage])
    edge_index_transformed = torch.tensor(edge_index_transformed, dtype=torch.long).T
    # create name for roc curve plot
    file_name = "Split-" + str(split_mode) + "/" + model.get_name() + "-" + str(model_id) + "_epoch-" + str(epoch) \
                + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # create roc plot
    accuracy_recommender.calc_ROC_curve(link_labels,
                                        link_logits,
                                        save_as_file=True,
                                        output_file_name=file_name)
    # calculate accuracy
    precision, recall = accuracy_recommender.accuracy_precision_recall(edge_index_transformed,
                                                                       link_labels,
                                                                       link_logits)
    # reset index of batcher
    batcher.reset_index()
    return precision, recall


def test_model_basic(
        model: GNN_GCNConv_homogen,
        batcher: edge_batch.EdgeConvolutionBatcher,
        device
):
    """
    Takes model and batcher executes all batches and accumulates the logits and labels to calulate and return the roc
    curve.

    Parameters
    ----------
    model : GNN_GCNConv_homogen
        model of the graph neural network
    batcher : edge_batch.EdgeConvolutionBatcher
        batcher that provides batch data objects to process
    device : torch.Device
        device to run the algorithm on

    Returns
    -------
    roc auc score of the tested edges
    """
    # poll the next element from the batcher
    current_batch, _ = batcher.next_element()
    # transfer data to device
    current_batch = current_batch.to(device)
    # create an empty list to put into the testors of the batches
    logits_list = []
    # for all batch objects in the batcher do
    while current_batch:
        # execute the test for the batches and append their logits and labels to the storage list
        logits_list.append((test_model_basic(model, current_batch), current_batch.y))
        # poll the next element from the stack
        current_batch, _ = batcher.next_element()
        # todo: should batch object be detached?
    # create the logits and label through concatenating the labels and logits from the batches
    logits = torch.cat([i[0] for i in logits_list])
    labels = torch.cat([i[1] for i in logits_list])
    # reset batcher index
    batcher.reset_index()
    # calculate roc auc score and return it
    return sklearn.metrics.roc_curve(labels.cpu(), logits.sigmoid().cpu())


def train_model(
        model: GNN_GCNConv_homogen,
        batch_list: edge_batch.EdgeConvolutionBatcher,
        optimizer,
        device
):
    """
    Execute the training for one epoch. Returns the averaged loss.

    Parameters
    ----------
    device : torch.Device
        device to run the algorithm on
    model : GNN_GCNConv_homogen
        model to train
    batch_list : edge_batch.EdgeConvolutionBatcher
        batcher from which to fetch the batch data objects
    optimizer
        Optimizer to use for training the model

    Returns
    -------
    loss : torch.Tensor
        averaged loss over all the batches using the edge_index size
    """
    # define accumulate variable for summing up loss from batches
    loss_accumulate = 0
    # poll a batch data object from the stack
    current_batch, _ = batch_list.next_element()
    # transfer data to device
    current_batch = current_batch.to(device)
    # variable for summing up total edge count
    total_edge_count = 0
    # for all batch data objects stored in batcher do
    while current_batch:
        # get the count of edges in the batch
        current_edge_count = current_batch.edge_index.size(1)
        # execute training for batch and sum loss up
        loss_accumulate += train_model_batch(model, optimizer, current_batch).detach() * current_edge_count
        # sum edge count up for total edge count
        total_edge_count += current_edge_count
        # poll next batch from batcher
        current_batch, _ = batch_list.next_element()
    # reset index
    batch_list.reset_index()
    # accumulate losses
    return loss_accumulate / total_edge_count
