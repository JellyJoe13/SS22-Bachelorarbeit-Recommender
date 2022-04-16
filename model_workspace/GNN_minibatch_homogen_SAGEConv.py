import torch.nn
import torch_geometric.nn.conv


class GNN_SAGEConv_homogen(torch.nn.Module):
    def __init__(
            self,
            num_features_input: int,
            num_features_out: int,
            num_features_hidden: int = 0
    ):
        self.mode = num_features_hidden > 0
        self.input_weighting = torch.nn.Linear(num_features_input, num_features_input)
        if self.mode:
            self.conv1 = torch_geometric.nn.conv.SAGEConv(num_features_input, num_features_hidden)
            self.conv2 = torch_geometric.nn.conv.SAGEConv(num_features_hidden, num_features_out)
        else:
            self.conv1 = torch_geometric.nn.conv.SAGEConv(num_features_input, num_features_out)
        self.bilinear = torch.nn.Bilinear(num_features_out, num_features_out, 1)
        self.flatten = torch.nn.Flatten(0, -1)

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index_input: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_weighting(x_input)
        x = self.conv1(x, pos_edge_index_input)
        if self.mode:
            x = self.conv2(x, pos_edge_index_input)
        x = self.bilinear(x[edge_index_input[0]], x[edge_index_input[1]])
        return self.flatten(x)

    def get_name(self) -> str:
        return "GNN_homogen_minibatch_SAGEConv_" + str(self.mode)
