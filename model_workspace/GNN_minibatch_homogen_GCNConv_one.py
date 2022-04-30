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

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index_input: torch.Tensor
    ) -> torch.Tensor:
        # push input data_related through Linear Layer that determines how important each feature of the node is
        x = self.init_linear(x_input)
        # run data_related and positive edge connectivity through convolution Layer
        x = self.conv(x, pos_edge_index_input)
        # convert the node embedding that was generated to the edge score/logits
        logits = (x[edge_index_input[0]] * x[edge_index_input[1]]).sum(dim=-1)
        return logits

    @staticmethod
    def get_name() -> str:
        return "GNN_homogen_minibatch_GCNConv_one"
