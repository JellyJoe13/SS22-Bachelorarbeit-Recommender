import typing

import pandas
import pandas as pd
import surprise
import torch_geometric.data
from surprise import Reader, Dataset, SVD
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import torch
from torch_geometric.data import Data
import rdkit
from rdkit.Chem import Descriptors, MolFromSmiles
import os
from os.path import exists
from x_data_transform import transform_and_scale_x_data


def pandas_to_GNN_pyg_edges(df, cid_translation_dictionary: dict, aid_translation_dictionary: dict):
    # definition of function used for creating the edge-set for test/train & active/inactive
    def sub_conv(activity_string: str, df_subset):
        # temp save selection set
        df_s = df_subset[df_subset.activity == activity_string]
        # determine how many entries the edge set will have. 2 edges per connection because of undirected edge
        count = df_s.size * 2
        # initialize the edges array
        edges = np.zeros(shape=(2, count))
        # create marker for current position in array
        marker = 0
        # iterate over the rows and enter the edges in the array
        for _, row in df_s.iterrows():
            # find mapped cid
            mol_id = cid_translation_dictionary[row.cid]
            # find mapped aid
            maid = aid_translation_dictionary[row.aid]
            # input one directed edge
            edges[0, marker] = mol_id
            edges[1, marker] = maid
            marker += 1
            # input other directed edge
            edges[0, marker] = maid
            edges[1, marker] = mol_id
            marker += 1
        # transform edges to torch object and return it
        return torch.tensor(edges, dtype=torch.long)

    # COMPUTE EDGE-SETS
    # generate edge-set for active/positive edges
    pos_edge_index = sub_conv('active', df)
    neg_edge_index = sub_conv('inactive', df)
    return pos_edge_index, neg_edge_index


def pandas_to_GNN_pyg_edges_v2(
        df: pandas.DataFrame,
        cid_translation_dictionary: dict,
        aid_translation_dictionary: dict
) -> torch.Tensor:
    """
    Function used for converting a pandas dataset to a pytorch-geometric GNN model.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe containing the data to transform
    cid_translation_dictionary : dict
        dictionary containing the mapping of cid's to new ids to use in the GNN
    aid_translation_dictionary : dict
        dictionary containing the mapping of aid's to new ids to use in the GNN

    Returns
    -------
    torch.tensor
        Tensor containing the transformed data - the edges of the GNN
    """
    # faster function to convert the pandas dataframe to GNN pytorch data
    # map ids to GNN id
    df['id_1'] = df['aid'].map(lambda x: aid_translation_dictionary[x])
    df['id_2'] = df['cid'].map(lambda x: cid_translation_dictionary[x])
    ret = {}
    for mode in ['active', 'inactive']:
        df_n = df[df.activity == mode]
        # extract edges information
        edge_direction_1 = df_n[['id_1', 'id_2']].to_numpy()
        edge_direction_2 = edge_direction_1.copy()
        # swap columns so that the other direction is simulated
        edge_direction_2[:, [0, 1]] = edge_direction_2[:, [1, 0]]
        # fuse both direction arrays
        ret[mode] = np.concatenate((edge_direction_1, edge_direction_2), axis=0)
    return torch.tensor(np.transpose(ret['active']), dtype=torch.long), torch.tensor(np.transpose(ret['inactive']),
                                                                                     dtype=torch.long)


def smiles_and_rdkit_chem_param_generation(
        df: pandas.DataFrame,
        aid_count: int,
        cid_count: int,
        cid_translation_dictionary: dict,
        generate: bool = True,
        empty_GNN_x: int = 0
) -> torch.Tensor:
    """
    Function used to generate the chemical descriptor data used as the node tensor in the GNN

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe : Input data which contains the SMILES data of the molecules to transform
    aid_count : int
        Integer telling the algorithm how many experiment ids there are; used for space allocation and indexing
    cid_count : int
        Integer telling the algorithm how many molecule ids there are; used for space allocation
    cid_translation_dictionary : dict
        Dictionary used for translating the compound ids to GNN ids
    generate : bool
        Truth value determining if or if not node data will be generated
    empty_GNN_x : int
        Integer specifying how many node features the ouput tensor should have if generate is set to False

    Returns
    -------
    torch.tensor
        Tensor containing the node data of the GNN
    """
    # simple check if empty_GNN_x is properly used
    assert empty_GNN_x >= 0
    # simple check for the aid and cid count variable
    assert aid_count > 0
    assert cid_count > 0
    # SEPARATING BETWEEN MODES missing
    if generate:
        # Modus where rdkit is used to generate the descriptors. any empty_GNN_x is ignored (currently)
        # Note: this is very time-consuming so that the pre-generated x is stored in a csv file in the data folder.
        # if the file exists we load it from there
        if exists("data/descriptors_x_transformed2.csv"):
            load_x = np.nan_to_num(np.loadtxt("data/descriptors_x_transformed2.csv", delimiter=","), nan=0)
            return torch.tensor(load_x, dtype=torch.float)
        # if the chemical descriptor file does not exist, generate and save it (as it takes 1h52 to compute it
        if exists("data/descriptors_x.csv"):
            # create x array
            x = np.zeros(shape=((aid_count + cid_count), len(Descriptors.descList)))
            # iterate over filtered and sorted table
            for _, row in df[['cid', 'smiles']].sort_values(by=['cid']).drop_duplicates(subset=['cid']).iterrows():
                # get corresponding id of cid
                mapped_id = cid_translation_dictionary[row.cid]
                # decode smiles notation to something the Descriptors can use using MolFrom Smiles
                mol = MolFromSmiles(row.smiles)
                # compute descriptors using mol and Descriptors.descList
                desc = np.array([func(mol) for _, func in Descriptors.descList])
                # put descriptors into the correct part of the x array
                x[mapped_id, :] = desc
            # x is now containing the chemical descriptors
            # transform and scale the chemical descriptors so that they can be used in the GNN
            # without generating NaN or Inf
            # values
            x = transform_and_scale_x_data(save_to_file=True,
                                           saving_path="data/descriptors_x_transformed2.csv",
                                           already_loaded_array=x)
            return torch.tensor(x, dtype=torch.float)
        else:
            # chemical descriptor csv exists but transformed version doesn't, so load it and save the generated array
            data = transform_and_scale_x_data(save_to_file=True,
                                              saving_path="data/descriptors_x_transformed2.csv",
                                              path="data/descriptors_x.csv")
            # turn it into a torch tensor and return it
            return torch.tensor(data, dtype=torch.float)
    else:
        # no descriptors will be computed
        if empty_GNN_x <= 0:
            # in this case the number of parameters shall be the same as the number of descriptors.
            # All values are set to 0
            return torch.tensor(np.ones(shape=((aid_count + cid_count), len(Descriptors.descList))), dtype=torch.float)
        else:
            # this means a specific amount is set, it will generate this specific number of parameters for each node
            return torch.tensor(np.ones(shape=((aid_count + cid_count), empty_GNN_x)), dtype=torch.float)


def data_transform_split(
        data_mode: int,
        split_mode: int = 0,
        path: str = "df_assay_entries.csv",
        empty_GNN_x: int = 0
) -> typing.Union[typing.Tuple[surprise.Trainset, typing.List[tuple]], torch_geometric.data.Data]:
    """
    A function that turns the pandas data into test and train-set data in which the mode determines which type of
    train test splitting is done.

    Parameters
    ----------
    data_mode : int
        defines if the desired output is a surprise data package (0) or the torch_geometric data
        (1 without rdkit information; 2 with)
    split_mode : str
        path and filename of the csv containing the chemistry dataset
    path : str
        determines which split mode is used: 0=random split entries, 1=molecule-wise, 2=assay-wise
    empty_GNN_x : int
        defines if data_mode==1 how many x-dimension each node should have in the pyg dataset

    Returns in case of data_mode == 0
    ---------------------------------
    trainset : surprise Trainset class
        Contains the data to train on
    testset : list of tuples with format (aid, cid, rating)
        Contains the data to test the Recommender algorithm on

    Returns in case of data_mode=1 or datamode=2
    --------------------------------------------
    data : torch_geometric.data.Data
        contains all train and test neg and pos edges plus x-parameter
    """
    # assert split_mode is within accepted range
    assert split_mode >= 0
    assert split_mode <= 2
    # assert data_mode is within accepted range
    assert data_mode >= 0
    assert data_mode <= 2
    # assert for empty_GNN_x
    assert empty_GNN_x >= 0
    # import data
    df = pd.read_csv(path)
    # define empty split variable for differing split types of groupwise and randomwise splitting
    split = None
    # separation of split methods
    if split_mode == 0:
        splitparam = df['cid'].to_numpy()
        split = ShuffleSplit(n_splits=1, random_state=0, test_size=0.2, train_size=None).split(splitparam, None)
    else:
        splitparam = None
        # mode 1 or 2 decides whether the split will be with cid or aid
        if split_mode == 1:
            splitparam = df['cid'].to_numpy()
        else:
            splitparam = df['aid'].to_numpy()
        # get the split test and train set as ids in numpy arrays
        split = GroupShuffleSplit(n_splits=1, random_state=0, test_size=0.2, train_size=None).split(splitparam, None,
                                                                                                    groups=splitparam)
    # unpack split index arrays from generator class in split
    test_ind = None
    train_ind = None
    for i, j in split:
        train_ind = i
        test_ind = j
    # now we have the indexes of the split data. Left to do is use this and create the data package of choice 
    if data_mode == 0:
        # data mode of surprise package
        # here we need to remodel the column activity to 0 and 1 boolean entries
        df['rating'] = df['activity'].map(lambda x: int(x == 'active'))
        # define reader to convert pandas dataframe to surprise package
        reader = Reader(rating_scale=(0, 1))
        # convert dataset importing only the entries from trainset index list using the iloc function
        trainset = Dataset.load_from_df(df.iloc[train_ind][['aid', 'cid', 'rating']], reader).build_full_trainset()
        testset = Dataset.load_from_df(df.iloc[test_ind][['aid', 'cid', 'rating']],
                                       reader).build_full_trainset().build_testset()
        return trainset, testset
    else:
        # build GNN edge set
        # we need to unify the cid and aid to one id set so that the ids for pytorch geometric are unique
        # ID TRANSLATION PART
        # count the number of aid's
        aid_count = np.unique(df['aid'].to_numpy()).shape[0]
        # count the number of cid's
        cid_count = np.unique(df['cid'].to_numpy()).shape[0]
        # create aid translation dictionary
        a = np.sort(np.unique(df['aid'].to_numpy()))
        a_n = np.arange(aid_count)
        aid_translation_dictionary = {a[i]: a_n[i] for i in range(aid_count)}
        # create cid translation dictionary
        c = np.sort(np.unique(df['cid'].to_numpy()))
        c_n = np.arange(aid_count, (aid_count + cid_count))
        cid_translation_dictionary = {c[i]: c_n[i] for i in range(cid_count)}
        # PROCESSING PART
        # the nodes in the graph are all the ids we have from aid and cid
        # the edges are the connections between aid and cid which are ACTIVE - these are stored in the pos edge_indexes,
        # the inactive edges are stored in the neg edge indexes
        # generate the edges of the trainset
        train_pos_edge_index, train_neg_edge_index = pandas_to_GNN_pyg_edges_v2(df.iloc[train_ind],
                                                                                cid_translation_dictionary,
                                                                                aid_translation_dictionary)
        # generate the edges of the testset
        test_pos_edge_index, test_neg_edge_index = pandas_to_GNN_pyg_edges_v2(df.iloc[test_ind],
                                                                              cid_translation_dictionary,
                                                                              aid_translation_dictionary)
        # call rdkit generating function with info if the x parameter should be empty or not
        x = smiles_and_rdkit_chem_param_generation(df, aid_count, cid_count, cid_translation_dictionary,
                                                   generate=(data_mode == 2), empty_GNN_x=empty_GNN_x)
        data = Data(x=x, train_pos_edge_index=train_pos_edge_index, train_neg_edge_index=train_neg_edge_index,
                    test_pos_edge_index=test_pos_edge_index, test_neg_edge_index=test_neg_edge_index)
        return data, aid_count
