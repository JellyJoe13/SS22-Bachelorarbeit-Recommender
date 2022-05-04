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
        """
        Initializing function for the graph neural network model

        Parameters
        ----------
        num_features_input : int
            number of features for each node input
        num_features_out : int
            number of parameters in the embedding for each node
        """
        super(GNN_GCNConv_homogen_basic, self).__init__()
        self.init_linear = torch.nn.Linear(num_features_input, num_features_input)
        self.conv = torch_geometric.nn.GCNConv(num_features_input, num_features_out)

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
        # push input data_related through Linear Layer that determines how important each feature of the node is
        x = self.init_linear(x_input)
        # run data_related and positive edge connectivity through convolution Layer
        x = self.conv(x, pos_edge_index_input)
        # convert the node embedding that was generated to the edge score/logits
        logits = (x[edge_index_input[0]] * x[edge_index_input[1]]).sum(dim=-1)
        return logits

    @staticmethod
    def get_name() -> str:
        """
        Return the name of the model which is used for naming of collected data.

        Returns
        -------
        name : str
            name of the model
        """
        return "GNN_homogen_minibatch_GCNConv_one"
