"""
Originally planned was a GIN convolution layer, however the convolution requires a torch.nn.Module parameter which
requires another initialization and configuration. SAGE is used as a replacement. Usage of GIN in further work.
"""
import torch.nn


class GNN_GINConv(torch.nn.Module):
    def __init__(
            self
    ):
        return None