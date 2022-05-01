import torch.nn
import torch_geometric.nn


class GNN_LGConv_homogen_variable(torch.nn.Module):
    """
    model using Light Graph Convolution for prediction
    """
    def __init__(
            self,
            input_x_features: int,
            number_convolutions: int = 1
    ):
        super(GNN_LGConv_homogen_variable, self).__init__()
        assert number_convolutions == 1 or number_convolutions == 2
        self.conv_mode = number_convolutions
        self.input_weighting = torch_geometric.nn.Linear(input_x_features, input_x_features)
        self.conv1 = torch_geometric.nn.conv.LGConv()
        if number_convolutions == 2:
            self.conv2 = torch_geometric.nn.conv.LGConv()

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index: torch.Tensor
    ) -> torch.Tensor:
        # input feature importance weighting
        x = self.input_weighting(x_input)
        # tunnel through convs
        x = self.conv1(x, pos_edge_index)
        if self.conv_mode == 2:
            x = self.conv2(x, pos_edge_index)
        # convert the node embedding that was generated to the edge score/logits
        logits = (x[edge_index_input[0]] * x[edge_index_input[1]]).sum(dim=-1)
        return logits

    def get_name(self) -> str:
        return "GNN_homogen_minibatch_LGConv_" + str(self.conv_mode)
