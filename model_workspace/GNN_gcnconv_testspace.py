import torch
import torch_geometric.data
from torch.nn import Bilinear, Flatten
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F


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
                    edge_index_input: torch.Tensor):
        x = self.init_linear(x_input)
        x = self.conv1(x, edge_index_input)  # first convolution layer
        x = x.relu()  # relu function for tu - disables negative values
        x = self.conv2(x, edge_index_input)  # second convolution layer
        # interpreting section
        x = self.bilinear(x[edge_index_input[0]], x[edge_index_input[1]])
        return self.endflatten(x)


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
        data: torch_geometric.data.Data,
        is_pos: bool,
        device
):
    # set the model to train mode
    model.train()
    # set the optimizer
    optimizer.zero_grad()
    # fit and predict the edges using the edge_index
    link_logits = model.fit_predict(data.x, data.edge_index)
    # get true predictions in form of a torch Tensor for loss computation
    link_labels = get_link_labels(data.edge_index, is_pos, device)
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
        data: torch_geometric.data.Data,
        is_pos: bool
):
    # put the model in evaluation mode
    model.eval()
    # call fit_predict though it will not be fitted due to the @torch.no_grad() annotation
    link_logits = model.fit_predict(data.x, data.edge_index)
    # generate the true label using the bool parameter
    link_label = get_link_labels(data.edge_index, is_pos)
    # return both of them (due to batch mode will be later on fused together
    return link_logits, link_label


# todo: weights for the batches so that positive learning will not be overruled?
# todo: add validation data
# todo: early stopping
# todo: add function to execute all batch for train and test and create the metrics (and save data for plotting)
