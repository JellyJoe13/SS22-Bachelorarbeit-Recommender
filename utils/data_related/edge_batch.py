import math

import torch_geometric.data
from torch_geometric.data import Data
import torch
import numpy as np
import typing


class EdgeConvolutionBatcher:
    """
    Class for batching based on edges and Convolution Layer.

    All edges will be exactly once be present in one batch data_related object, nodes may appear multiple times. Used for edge
    prediction or learning that uses Convolution layer.
    """

    def __init__(
            self,
            data: Data,
            edge_sample_count: int = 1000,
            convolution_depth: int = 2,
            convolution_neighbor_count: int = 100,
            is_directed: bool = False,
            train_test_identifier: str = "train"
    ):
        """
        Initializes and creates the EdgeConvolutionBatcher with its parameters/settings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data object which content will be used for batching
        edge_sample_count : int
            Specifies the number of edges per batch - excluding the edges of convolution sampling
        convolution_depth : int
            Specifies the depth of neighborhood from which edges will be sampled with convolution sampling
        convolution_neighbor_count : int
            Specifies the number of edges to be sampled from a node while convolution sampling
        is_directed : bool
            Specifies if the input data_related is an undirected graph (2 edges for each connection between nodes)
        train_test_identifier : str
            Specifies if the train or test edges of the dataset should be batched
        """
        self.edge_sample_count = edge_sample_count
        self.convolution_depth = convolution_depth
        self.convolution_neighbor_count = convolution_neighbor_count
        self.original_data = data
        # parameters for iterating through it or storing the separated list
        self.batch_list = None
        self.batch_index = 0
        self.is_directed = is_directed
        self.mode_identifier = train_test_identifier
        pos_count = self.original_data[self.mode_identifier + "_pos_edge_index"].size(1)
        neg_count = self.original_data[self.mode_identifier + "_neg_edge_index"].size(1)
        self.edges = torch.cat([self.original_data[self.mode_identifier + "_pos_edge_index"],
                                self.original_data[self.mode_identifier + "_neg_edge_index"]],
                               dim=-1)
        self.target = torch.cat([torch.ones(pos_count, dtype=torch.float), torch.zeros(neg_count, dtype=torch.float)])
        # sort edges
        _, idx_sort = self.edges[0].sort(dim=-1)
        self.edges = self.edges[:, idx_sort]
        self.target = self.target[idx_sort]

    def reset_index(self) -> None:
        """
        Function for resetting to the beginning initializing a new epoch in most cases.

        Returns
        -------
        Nothing
        """
        self.batch_index = 0
        return

    # define following function for removing duplicate edges (bidirectional)
    @staticmethod
    def remove_undirected_duplicate_edge(edge_index_local: torch.Tensor):
        """
        Function used for removing duplicate edges (if edge (x,y) also exists as (y,x) in graph) which could be better
        to include in splitting as required node per split can be reduced.

        Parameters
        ----------
        edge_index_local : torch.Tensor
            edge index to search for duplicate entries in

        Returns
        -------
        torch.Tensor
            edge index which was input but without the edges but without duplicate edges
        """
        return edge_index_local[:, edge_index_local[0] < edge_index_local[1]]

    def do_batch_split(
            self
    ) -> None:
        """
        Function for splitting the input data_related in batch parts.

        Does not return anything and will be automatically called in the function next_element().

        Returns
        -------
        Nothing
        """
        # batch list containing the data_related object batch elements
        batch_list = []
        # differ if we assume an undirected graph or a directed one
        if self.is_directed:
            # DATA LOADING AND PREPARATION SECTION
            # concatenate edge index and write y labels accordingly
            edge_index = torch.cat([self.original_data[self.mode_identifier + "_pos_edge_index"],
                                    self.original_data[self.mode_identifier + "_neg_edge_index"]],
                                   dim=-1)
            y = torch.zeros(
                edge_index.size(1),
                dtype=torch.float)
            y[:self.original_data[self.mode_identifier + "_pos_edge_index"]] = 1
            # sort edges
            idx_sort = edge_index[0].sort(dim=-1)
            edge_index = edge_index[:, idx_sort]
            y = y[:, idx_sort]
            # EDGE SPLITTING SET SECTION
            # if it can be assumed that we do have a directed graph, not an undirected one (meaning 2 directed edges for
            # each edge
            for i in range(0, edge_index.size(1), self.edge_sample_count):
                # select range of edge_indexes
                selected_edge_index = edge_index[:, i:(i + self.edge_sample_count)]
                # select range of y
                selected_y = y[i:(i + self.edge_sample_count)]
                # sampling positive edges is not done here but when pulling the next batch with next() function
                # put selected data_related into batch list
                batch_list.append((selected_edge_index, selected_y))
        else:
            # create/transform pos section
            edge_index = self.remove_undirected_duplicate_edge(
                self.original_data[self.mode_identifier + "_pos_edge_index"])
            # create/transform neg section
            neg_edge_index = self.remove_undirected_duplicate_edge(
                self.original_data[self.mode_identifier + "_neg_edge_index"])
            # create y
            y = torch.zeros(
                edge_index.size(1) + neg_edge_index.size(1),
                dtype=torch.float)
            y[:edge_index.size(1)] = 1
            # fuze pos and neg edge index together
            edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            # sort edges
            idx_sort = edge_index[0].sort(dim=-1)
            edge_index = edge_index[:, idx_sort]
            y = y[:, idx_sort]
            # split edges (only half sample size as each edge exists 2 times in the returned data_related object
            for i in range(0, edge_index.size(1), int(self.edge_sample_count / 2)):
                # selected range of indexes
                selected_edge_index = edge_index[:, i:(i + int(self.edge_sample_count / 2))]
                # selected range of y
                selected_y = y[i:(i + int(self.edge_sample_count / 2))]
                # sampling done when retrieving batch
                # append this batch of edges and labels to batch list
                batch_list.append((selected_edge_index, selected_y))
        self.batch_list = batch_list
        return

    def neighbor_sampling(
            self,
            node_list: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that executes neighbor sampling of positive edges of given nodes of the dataset.

        Does not compute full neighbor sampling of all edges adjacent to the given nodes and also not more that depth 1.
        If more that depth 1 should be sampled re-execute the function several times with supplying the returned node id
        list or consider using the NeighborLoader/NeighborSampler function of pytorch geometric.

        Parameters
        ----------
        node_list : torch.Tensor
            List of nodes from which the neighbor sampling step will be executed.

        Returns
        -------
        edge_sample : typing.List[torch.Tensor]
            Tensor containing the sampled edges while convolution sampling of positive edges
        new_node_list : torch.Tensor
            Tensor containing the new list of nodes used after sampling
        """
        edge_sample = []
        new_nodes = []
        # filter pos edge index after incident edges and choose a few random ones (parameter controlled)
        for node in node_list:
            # filter edges incident to this node
            selected_edges = self.original_data[self.mode_identifier + "_pos_edge_index"][
                1, self.original_data[self.mode_identifier + "_pos_edge_index"][0] == node]
            # randomize the samples
            np.random.shuffle(selected_edges.detach().numpy())
            # choose the first self.edge_sample_count edges and put them into the list
            selected_edges = selected_edges[:self.convolution_neighbor_count]
            # append to list
            new_nodes.append(selected_edges)
            # reshape selected_edges
            selected_edges = selected_edges.reshape((selected_edges.size(0), 1))
            # add edges to edge_list
            selected_edges = torch.cat([selected_edges.new_full(selected_edges.size(), node),
                                        selected_edges], dim=-1)
            edge_sample.append(selected_edges.t())
        # compute a list of new nodes contained in the batch and make it unique
        new_node_list = torch.cat(new_nodes).unique()
        # return both results
        return edge_sample, new_node_list

    def next_element(self):
        # get start and end position and check logic to stop
        idx_start = self.edge_sample_count * self.batch_index
        idx_end = self.edge_sample_count * (self.batch_index + 1)
        if idx_end > self.edges.size(1):
            idx_end = self.edges.size(1)
        elif idx_start > self.edges.size(1):
            return None
        # get working set of y and edges
        current_y = self.target[idx_start:idx_end]
        current_e = self.edges[:, idx_start:idx_end].clone()
        # get the unique node ids
        node_ids = current_e.flatten().unique()
        # storage for collected edges and nodes from neighborsampling
        edge_collector = []
        node_collector = [node_ids]
        # run neighborhood sampling
        c = node_ids
        for i in range(self.convolution_depth):
            # execute one iteration of neighbor sampling
            e, n = self.neighbor_sampling(c)
            c = n
            # append generated info to collectors
            node_collector.append(n)
            edge_collector += e
        # fuze collectors
        edge_collector = torch.cat(edge_collector, dim=-1)
        node_collector = torch.cat(node_collector).unique()
        # create translation dictionary
        translation_dict = dict(zip(node_collector, np.arange(node_collector.size(0))))
        # translate edges
        for i in translation_dict:
            value = translation_dict[i]
            edge_collector[edge_collector == i] = value
            current_e[current_e == i] = value
        # create data object
        data = Data(x=self.original_data.x[node_collector],
                    edge_index=current_e,
                    pos_edge_index=edge_collector,
                    y=current_y)
        # create reverse dict
        reverse_dict = dict(zip(translation_dict.values(), translation_dict.keys()))
        # increase batch index
        self.batch_index += 1
        return data, reverse_dict

    def next_element_deprecated(self) -> typing.Tuple[torch_geometric.data.Data, dict]:
        """
        Function used for iterating through the split dataset. Computes random neighbor sampling while loading batch
        data_related object.

        Returns
        -------
        data_related : torch_geometric.data.Data
            Data object containing the batched fragment of the original input data_related
        """
        # if the batch split has not been yet calulated, calculate it
        if not self.batch_list:
            self.do_batch_split()
        # check if we are still in range of the batch_list
        if not self.batch_index < len(self.batch_list):
            # we are out of boundaries, return None
            return None
        # we are in range of batch_list
        current_edges = self.batch_list[self.batch_index][0]
        current_y = self.batch_list[self.batch_index][1]
        # calculate the unique nodes which will be sampled
        node_ids = current_edges.flatten().unique()
        # increment the batch index
        self.batch_index += 1
        # NEIGHBOR SAMPLING SECTION
        # self.convolution_depth time we fetch a number of edges surrounding the nodes of the edges fetched before
        convolution_edges = []
        for depth in self.convolution_depth:
            temp_edges, node_ids = self.neighbor_sampling(node_ids)
            convolution_edges.append(temp_edges)
        # fuze edges
        convolution_edges = torch.cat(convolution_edges, dim=-1)
        # create translation dictionary
        translation_dict = dict(zip(node_ids, np.arange(node_ids.size(0))))
        # TRANSLATION SECTION
        # translate batch edges
        for i in range(current_edges.size(0)):
            for j in range(current_edges.size(1)):
                current_edges[i, j] = translation_dict[current_edges[i, j]]
        # translate convolution_edges
        for i in range(convolution_edges.size(0)):
            for j in range(convolution_edges.size(1)):
                convolution_edges[i, j] = translation_dict[convolution_edges[i, j]]
        # build Data object
        data = Data(x=self.original_data.x[node_ids],
                    edge_index=current_edges,
                    pos_edge_index=convolution_edges,
                    y=current_y)
        # invert translation dictionary to be able to turn the ids back to their original forms
        reverse_dict = {v: k for k, v in translation_dict}
        return data, reverse_dict

    def get_element(
            self,
            index: int
    ) -> torch_geometric.data.Data:
        """
        Grant index wise access to batch objects stored in this class. Uses function next_element in order to achieve
        this currently.

        Parameters
        ----------
        index : int
            index of the batch data_related object to fetch/compute.

        Returns
        -------
        torch.data_related.Data
            batch data_related object that was requested
        """
        # save the previous index
        previous_index = self.batch_index
        # set the new index
        self.batch_index = index
        # compute element
        data = self.next_element()
        # reset the index to old value
        self.batch_index = previous_index
        # return computed batch data_related object
        return data


class EdgeBatcher:
    def __init__(
            self,
            data: torch_geometric.data.Data,
            num_selection_edges: int = 10000,
            mode: str = "train"
    ):
        """
        Initialize EdgeBatcher.

        Parameters
        ----------
        data : torch_geometric.data.Data
            original data on which the splitting operation is to be performed
        num_selection_edges : int
            number of edges that will be in the subset data object which will be predicted
        mode : str
            determines if the split shall be done on the trainset, testset or validation set
        """
        # set parameters of class
        self.original_data = data
        self.mode_identifier = mode
        self.num_selection_edges = num_selection_edges
        # get count of positive and negative edges
        pos_count = self.original_data[self.mode_identifier + "_pos_edge_index"].size(1)
        neg_count = self.original_data[self.mode_identifier + "_neg_edge_index"].size(1)
        # create subset basic information
        self.edges = torch.cat([self.original_data[self.mode_identifier + "_pos_edge_index"],
                                self.original_data[self.mode_identifier + "_neg_edge_index"]],
                               dim=-1)
        self.target = torch.cat([torch.ones(pos_count, dtype=torch.float), torch.zeros(neg_count, dtype=torch.float)])
        # create randomize index
        index = np.arange(pos_count+neg_count)
        np.random.shuffle(index)
        # randomize edges with index
        self.edges = self.edges[:, index]
        self.target = self.target[:, index]
        # calculate number of subsets
        self.len = math.ceil((pos_count + neg_count) / num_selection_edges)

    def __len__(self) -> int:
        """
        Return the number of all subsets/batch objects.

        Returns
        -------
        int
            number of subsets
        """
        return self.len

    def __call__(
            self,
            i: int
    ) -> torch_geometric.data.Data:
        """
        Get the i-th subset of the batcher.

        Parameters
        ----------
        i : int
            specifies index of batch data object to fetch and construct a subset data object for.

        Returns
        -------
        batch_data : torch_geometric.data.Data
            batch data object which is a subset of the original data set
        """
        # ensure proper range for i
        if i < 0:
            return None
        elif i >= self.len:
            return None
        # determine start and end
        idx_start = self.num_selection_edges * i
        idx_end = self.num_selection_edges * (i + 1)
        if idx_end > self.edges.size(1):
            idx_end = self.edges.size(1)
        # return selected subset
        data = torch_geometric.data.Data(
            x=self.original_data.x,
            pos_edge_index=self.original_data[self.mode_identifier + "_pos_edge_index"],
            edge_index=self.edges[:, idx_start:idx_end],
            y=self.target[idx_start:idx_end]
        )
        return data

    def randomize(self) -> None:
        """
        Randomize subsets.

        Returns
        -------
        Nothing.
        """
        # create randomize index
        index = np.arange(self.edges.size(1))
        np.random.shuffle(index)
        # randomize edges with index
        self.edges = self.edges[:, index]
        self.target = self.target[:, index]
        return
