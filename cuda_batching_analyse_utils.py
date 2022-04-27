import typing

import torch
import torch_geometric.data
import torch_geometric.nn.functional as F


def max_subset_size(
        model,
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.binary_cross_entropy_with_logits
) -> None:
    assert torch.cuda.is_available()
    num_nodes = 457560
    num_node_features = 205
    x = torch.rand(num_nodes, num_node_features)
    num_train_pos_edges = 979616
    pos_edge_index = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, num_train_pos_edges)
    )
    current_batch_size = 100000
    # define device and optimizer
    device = torch.device('cuda')
    optimizer = torch.optim.Adam(params=model.parameters())
    # transfer model to cuda
    model = model.to()
    while True:
        print("Attempting batch size", current_batch_size)
        data = torch_geometric.data.Data(
            x=x,
            pos_edge_index=pos_edge_index,
            edge_index=torch.randint(
                low=0,
                high=num_nodes,
                size=(2, current_batch_size)
            ),
            y=torch.randint(
                low=0,
                high=2,
                size=(current_batch_size,)
            )
        )
        data = data.to(device)
        model.train()
        optimizer.zero_grad()
        link_logits = model.fit_predict(data.x, data.edge_index, data.pos_edge_index)
        loss = loss_function(link_logits, data.y)
        loss.backward()
        optimizer.step()
        data = data.detach()
        print("confirmed")
        current_batch_size *= 2
    return
