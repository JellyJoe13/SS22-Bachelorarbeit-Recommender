import torch.nn
import torch_geometric.nn.conv
from torch_geometric.nn import Linear


class GNN_GATConv_homogen(torch.nn.Module):
    def __init__(
            self,
            num_features_input: int,
            num_features_out: int,
            num_features_hidden: int = 0
    ):
        """
        Initializing function for the graph neural network model

        Parameters
        ----------
        num_features_input : int
            number of features for each node input
        num_features_out : int
            number of parameters in the embedding for each node
        num_features_hidden : int
            number of hidden features for each node
        """
        super(GNN_GATConv_homogen, self).__init__()
        self.mode = num_features_hidden > 0
        self.input_weighting = Linear(num_features_input, num_features_input)
        if num_features_hidden > 0:
            self.conv1 = torch_geometric.nn.conv.GATConv(num_features_input, num_features_hidden)
            self.conv2 = torch_geometric.nn.conv.GATConv(num_features_hidden, num_features_out)
        else:
            self.conv1 = torch_geometric.nn.conv.GATConv(num_features_input, num_features_out)

    def fit_predict(
            self,
            x_input: torch.Tensor,
            edge_index_input: torch.Tensor,
            pos_edge_index_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Function for generating the prediction for the input data for this model.

        Parameters
        ----------
        x_input : torch.Tensor
            input node feature data
        edge_index_input : torch.Tensor
            input edge information on the edges to predict
        pos_edge_index_input : torch.Tensor
            input information on the positive edges which are used to generate the node embedding

        Returns
        -------
        logits : torch.Tensor
            Prediction/Score for the edges to predict
        """
        x = self.input_weighting(x_input)
        x = self.conv1(x, pos_edge_index_input)
        if self.mode:
            x = self.conv2(x, pos_edge_index_input)
        # convert the node embedding that was generated to the edge score/logits
        logits = (x[edge_index_input[0]] * x[edge_index_input[1]]).sum(dim=-1)
        return logits

    def get_name(self) -> str:
        """
        Return the name of the model which is used for naming of collected data.

        Returns
        -------
        name : str
            name of the model
        """
        return "GNN_homogen_minibatch_GATConv_" + str(self.mode)
