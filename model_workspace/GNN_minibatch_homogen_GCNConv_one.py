import torch.nn
import torch_geometric.nn


class GNN_GCNConv_homogen_basic(torch.nn.Module):
    """
    Simple model containing only one Convolution Layer but same Linear layers as the advanced model
    """
    def __init__(
            self,
            num_features_input: int,
            num_features_out: int
    ):
        super(GNN_GCNConv_homogen_basic, self).__init__()
        self.init_linear = torch.nn.Linear(num_features_input, num_features_input)
        self.conv = torch_geometric.nn.GCNConv(num_features_input, num_features_out)
        self.bilinear = torch.nn.Bilinear(num_features_out, num_features_out, 1)
        self.endflatten = torch.nn.Flatten(0, -1)

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index_input: torch.Tensor
    ) -> torch.Tensor:
        # push input data through Linear Layer that determines how important each feature of the node is
        x = self.init_linear(x_input)
        # run data and positive edge connectivity through convolution Layer
        x = self.conv(x, pos_edge_index_input)
        # convert the node embedding that was generated to the edge score/logits
        x = self.bilinear(x[edge_index_input[0]], x[edge_index_input[1]])
        # flatten the input to dimension 1 and output it
        return self.endflatten(x)

    @staticmethod
    def get_name() -> str:
        return "GNN_homogen_minibatch_GCNConv_one"
