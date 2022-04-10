import sklearn.metrics
import torch
import torch_geometric.data
from torch.nn import Bilinear, Flatten
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F
from datetime import datetime

import accuracy_recommender
import edge_batch


class GNN_GCNConv_homogen(torch.nn.Module):
    """
    batch version implementation of GNN using all edges for input in GCNConv as batching positive and negative together
    will not work without rewriting the Neighborloader Class, so we are learning from both to predict.
    Inspired by content shown in https://antoniolonga.github.io/Pytorch_geometric_tutorials/posts/post12.html
    from Antionio Longa
    """

    def __init__(self,
                 num_features: int):
        super(GNN_GCNConv_homogen, self).__init__()
        self.init_linear = Linear(num_features, num_features)
        self.conv1 = GCNConv(num_features, int(num_features / 2))
        self.conv2 = GCNConv(int(num_features / 2), 64)
        self.bilinear = Bilinear(64, 64, 1)
        self.endflatten = Flatten(0, -1)

    def fit_predict(self,
                    x_input: torch.Tensor,
                    edge_index_input: torch.Tensor,
                    pos_edge_index_input: torch.Tensor
                    ):
        x = self.init_linear(x_input)
        x = self.conv1(x, pos_edge_index_input)  # first convolution layer
        x = x.relu()  # relu function for tu - disables negative values
        x = self.conv2(x, pos_edge_index_input)  # second convolution layer
        # interpreting section
        x = self.bilinear(x[edge_index_input[0]], x[edge_index_input[1]])
        return self.endflatten(x)

    @staticmethod
    def get_name() -> str:
        """
        Defines a name of the model that is used if an output file is generated.

        Returns
        -------
        str
            Name of the model, used in output files
        """
        return "GNN_GCNConv_homogen_minibatch"


# unused as y labels will be given through y
def get_link_labels(
        edge_index: torch.Tensor,
        is_pos: bool,
        device_to_run
) -> torch.Tensor:
    if is_pos:
        return torch.ones(edge_index.size(1), device=device_to_run)
    else:
        return torch.zeros(edge_index.size(1), device=device_to_run)


def train_model_batch(
        model: GNN_GCNConv_homogen,
        optimizer,
        data: torch_geometric.data.Data
):
    # set the model to train mode
    model.train()
    # set the optimizer
    optimizer.zero_grad()
    # fit and predict the edges using the edge_index
    link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
    # get true predictions in form of a torch Tensor for loss computation
    link_labels = data.y
    # calculate loss
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
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
    # put the model in evaluation mode
    model.eval()
    # call fit_predict though it will not be fitted due to the @torch.no_grad() annotation
    link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
    return link_logits


def test_model_advanced(
        model: GNN_GCNConv_homogen,
        batcher: edge_batch.EdgeConvolutionBatcher,
        id_breakpoint: int,
        epoch: int = 0
) -> tuple(int, int):
    current_batch, retranslation_dict = batcher.next_element()
    batch_loop_storage = []
    edge_index_transformed = []
    while current_batch:
        link_logits = test_model_basic(model, current_batch)
        batch_loop_storage.appen((current_batch.y,
                                  link_logits))
        current_batch, retranslation_dict = batcher.next_element()
        for edge in current_batch.edge_index.T:
            edge_index_transformed.append([retranslation_dict[edge[0]], retranslation_dict[edge[1]]])
    link_logits = torch.cat([i[0] for i in batch_loop_storage])
    link_labels = torch.cat([i[1] for i in batch_loop_storage])
    edge_index_transformed = torch.tensor(edge_index_transformed, dtype=torch.long).T
    # create name for roc curve plot
    file_name = model.get_name()+"_epoch-"+str(epoch)+"_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # create roc plot
    accuracy_recommender.calc_ROC_curve(link_labels,
                                        link_logits,
                                        save_as_file=True,
                                        output_file_name=file_name)
    # calculate accuracy
    precision, recall = accuracy_recommender.accuracy_precision_recall(edge_index_transformed,
                                                                       link_labels,
                                                                       link_logits,
                                                                       id_breakpoint)
    return precision, recall


# todo: add validation data
# todo: early stopping

def test_model_basic(
        model: GNN_GCNConv_homogen,
        batcher: edge_batch.EdgeConvolutionBatcher
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

    Returns
    -------
    roc auc score of the tested edges
    """
    # poll the next element from the batcher
    current_batch = batcher.next_element()
    # create an empty list to put into the testors of the batches
    logits_list = []
    # for all batch objects in the batcher do
    while current_batch:
        # execute the test for the the batches and append their logits and labels to the storage list
        logits_list.append((test_model_basic(model, current_batch), current_batch.y))
        # poll the next element from the stack
        current_batch = batcher.next_element()
    # create the logits and label through concatenating the labels and logits from the batches
    logits = torch.cat([i[0] for i in logits_list])
    labels = torch.cat([i[1] for i in logits_list])
    # calculate roc auc score and return it
    return sklearn.metrics.roc_curve(labels.cpu(), logits.sigmoid().cpu())


def train_model(
        model: GNN_GCNConv_homogen,
        batch_list: edge_batch.EdgeConvolutionBatcher,
        optimizer
):
    """
    Execute the training for one epoch. Returns the averaged loss.

    Parameters
    ----------
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
    current_batch = batch_list.next_element()
    # variable for summing up total edge count
    total_edge_count = 0
    # for all batch data objects stored in batcher do
    while current_batch:
        # get the cound of edges in the batch
        current_edge_count = current_batch.edge_index.size(1)
        # execute training for batch and sum loss up
        loss_accumulate += train_model_batch(model, optimizer, current_batch).detach() * current_edge_count
        # sum edge count up for total edge count
        total_edge_count += current_edge_count
        # poll next batch from batcher
        current_batch = batch_list.next_element()
    # accumulate losses
    return loss_accumulate / total_edge_count
