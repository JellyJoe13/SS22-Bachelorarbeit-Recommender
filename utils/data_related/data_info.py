import torch
import numpy as np
import pandas as pd


class DataInfoHandler:
    def __init__(
            self,
            id_breakpoint,
            molecule_dict,
            experiment_dict
    ):
        """
        Initializing function of DataInfoHandler.

        Parameters
        ----------
        id_breakpoint : int
            breakpoint of GNN ids which separate molecules and experiment in homogeneous graph
        molecule_dict : dict
            dictionary containing information on how cid are mapped to GNN_ids
        experiment_dict : dict
            dictionary containing information on how aid are mapped to GNN_ids
        """
        self.id_breakpoint = id_breakpoint
        self.molecule_dict = molecule_dict
        self.experiment_dict = experiment_dict

    def create_mapped_edges(
            self,
            cid: np.ndarray,
            aid: np.ndarray
    ) -> torch.Tensor:
        """
        Take input id of molecules and experiments and map and convert information to pytorch tensor containing the
        edges.

        Parameters
        ----------
        cid : np.ndarray
            numpy array containing the ids of the molecules to map
        aid : np.ndarray
            numpy array containing the ids of the experiments to map

        Returns
        -------
        edges : torch.Tensor
            tensor containing the edge index of the input information on molecule-experiment pairs
        """
        df = pd.DataFrame(data={
            "aid": aid,
            "cid": cid
        })
        df["aid"] = df["aid"].map(lambda x: self.experiment_dict[x])
        df["cid"] = df["cid"].map(lambda x: self.molecule_dict[x])
        # create fused numpy array
        edges = df[["aid", "cid"]].to_numpy().T
        # to torch tensor
        edges = torch.tensor(edges, dtype=torch.long)
        # return mapped and converted edges
        return edges
