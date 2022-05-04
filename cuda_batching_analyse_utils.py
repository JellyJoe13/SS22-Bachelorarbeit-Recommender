import typing

import torch
import torch_geometric.data
import torch.nn.functional as F
from utils.accuracy.accuarcy_bpr import binary_loss_adapter


def max_subset_size(
        model,
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = binary_loss_adapter,
        known_prev_max: int = None,
        start_size: int = 100000
) -> None:
    """
    Function that is used to determine the max size that CUDA can handle regarding the batching size/number of edges
    to predict/learn on.

    Parameters
    ----------
    model
        Model used for learning/predicting edges
    loss_function : typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        loss function used for training the model
    known_prev_max : int
        parameter defining if a previous run with this tool was executed and what the largest confirmed size was so that
        the increase is not exponential but linear.
    start_size : int
        define with with size the tool with start with for the exponential testing of batch size

    Returns
    -------
    None
    """
    assert torch.cuda.is_available()
    num_nodes = 457560
    num_node_features = 205
    x = torch.rand(num_nodes, num_node_features)
    num_train_pos_edges = 979616
    pos_edge_index = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, num_train_pos_edges),
        dtype=torch.long
    )
    current_batch_size = start_size if not known_prev_max else known_prev_max
    # define device and optimizer
    device = torch.device('cuda')
    optimizer = torch.optim.Adam(params=model.parameters())
    # transfer model to cuda
    model = model.to(device)
    while True:
        print("Attempting batch size", current_batch_size)
        data = torch_geometric.data.Data(
            x=x,
            pos_edge_index=pos_edge_index,
            edge_index=torch.randint(
                low=0,
                high=num_nodes,
                size=(2, current_batch_size),
                dtype=torch.long
            ),
            y=torch.randint(
                low=0,
                high=2,
                size=(current_batch_size,),
                dtype=torch.float
            )
        )
        data = data.to(device)
        model.train()
        optimizer.zero_grad()
        link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
        loss = loss_function(link_logits, data.y, data.edge_index)
        loss.backward()
        optimizer.step()
        data = data.detach()
        print("confirmed")
        if not known_prev_max:
            current_batch_size *= 2
        else:
            current_batch_size += int(known_prev_max/10)
    return
