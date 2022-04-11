import torch
from torch.nn import Bilinear, Flatten
from torch_geometric.nn import GCNConv, Linear


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

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Fit and predict function of the graph neural network predictor. Calculates a node embedding using the node data
        and positive edges in the graph and then uses this embedding to calculate the logits.

        Parameters
        ----------
        x_input : torch.Tensor
            tensor containing the data of the nodes
        edge_index_input : torch.Tensor
            tensor containing the edges to predict
        pos_edge_index_input : torch.Tensor
            tensor containing the positive edges used for calculating the node embedding

        Returns
        -------
        logits : torch.Tensor
            predicted logits of the edges. will be put through sigmoid() function to get link probabilities
        """
        # input linear layer to learn the importance of the different chemical Descriptors or filter out the harmful
        # ones
        x = self.init_linear(x_input)
        # first convolution layer - depth one
        x = self.conv1(x, pos_edge_index_input)
        # relu function for disabling negative values - GCNConv cannot handle them
        x = x.relu()
        # second convolution layer - depth two
        x = self.conv2(x, pos_edge_index_input)
        # interpreting section, similar to vector product but with weighting and possible bias. learnable parameters
        x = self.bilinear(x[edge_index_input[0]], x[edge_index_input[1]])
        # return the logits
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
    # deprecation warning
    print("========================================================================")
    print("Warning: This function is marked as deprecated and will be removed soon.")
    print("========================================================================")
    if is_pos:
        return torch.ones(edge_index.size(1), device=device_to_run)
    else:
        return torch.zeros(edge_index.size(1), device=device_to_run)
