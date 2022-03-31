import math

import numpy as np


def transform_and_scale_x_data(
        save_to_file: bool = False,
        path: str = "data/descriptors_x.csv",
        saving_path: str = "data/descriptors_x_transformed2.csv",
        already_loaded_array=None
) -> np.ndarray:
    """
    This function takes the original data from rdkit containing the chemical descriptors and transforms them.

    Parameters
    ----------
    save_to_file
        bool : determines if a file should be generated for the transformed and scaled data
    path
        str : used for specifying the input chemical parameters to load
    saving_path
        str : used for specifying the file the data should be written to in case save_to_file is True
    already_loaded_array
        np.ndarray : if the dataset has already been loaded, use it instead of loading it from anew

    Returns
    -------
    numpy.ndarray containing the transformed and scaled data
    """
    load_x = None
    # determine if data needs to be loaded from file or has been provided in parameter
    if already_loaded_array:
        # set load_x to input array
        load_x = already_loaded_array.copy()
    else:
        # load the numpy array from existing file
        load_x = np.nan_to_num(np.loadtxt(path, delimiter=","), nan=0).astype('float64')
    # ZERO COLUMN SECTION
    # find the columns which only contain zeros and delete them
    index_cols = np.argwhere(np.all(load_x == 0, axis=0))
    load_x = np.delete(load_x, index_cols, axis=1)
    # NEGATIVE ENTRY DEALING SECTION
    # next is how to deal with the negative values
    # we will shift them so that they are positive. Even if there are very large negative entries they will be either
    # scaled or log treated, which will keep the relationship
    # finding the column with one or more negative entries
    index_cols = (load_x < 0).sum(axis=0) > 0
    # determine the minimum for shifting and altering it so that the smallest entry does not get 0 but 0.1
    min_entries = load_x[:, index_cols].min(axis=0) - 0.1
    # shift the negative columns
    load_x[:, index_cols] -= np.transpose(np.repeat(np.reshape(min_entries, (-1, 1)), load_x.shape[0], axis=1))
    # IMMENSELY LARGE ENTRIES HANDLING
    # Large entries (>10^10) are a problem as they make the other entries too small when the column is scaled by them.
    # For this reason we have the following rules: If we have 5 or more entries above 10^10 we keep them and apply
    # logarithmic scaling (base 10). Else we assume that they are outliers and delete them.
    # compute the columns that have 5 or more very large entries
    index_cols = (load_x > math.pow(10, 10)).sum(axis=0) >= 5
    # for these rows apply logarithmic scaling (we add 1 because we do not want to convert 0 to -inf)
    load_x[:, index_cols] = np.log10(load_x[:, index_cols] + 1)
    # now the remaining entries larger than 10^10 are outlier and set to 0
    load_x[load_x > math.pow(10, 10)] = 0
    # DELETE ZERO COLUMNS IN CASE WE GENERATED ZERO ROWS WITH OUTLIER REMOVAL
    load_x = np.delete(load_x, np.argwhere(np.all(load_x == 0, axis=0)), axis=1)
    # SCALING SECTION
    # each entry should now be brought between 0 and 1
    # scale matrix using the column wise maximum
    load_x /= np.max(load_x, axis=0)
    # ZERO ROW HANDLING SECTION
    # in case we encounter zero rows (if we do not offset negative entries) we add a column with random entries
    load_x = np.c_[load_x, np.random.rand(load_x.shape[0])]
    # load_x is finished and ready to be returned, checking data and optional saving is following
    # DATA CHECKUP SECTION
    # do we have negative entries?
    assert (load_x < 0).sum() == 0
    # do we have entries larger than 1 (scaling broke down somewhere
    assert (load_x > 1).sum() == 0
    # do we have columns that only contain zeros?
    assert ((load_x == 0).sum(axis=0) < load_x.shape[0]).all()
    # do we have zero rows?
    assert ((load_x == 0).sum(axis=1) < load_x.shape[1]).all()
    # SAVING TO FILE SECTION
    # parameter controlled
    if save_to_file:
        # if the parameter is True this means the transformed data should be written to a file
        # save the file
        np.savetxt(saving_path, load_x, delimiter=",")
    # RETURN SECTION
    # returning the transformed and scaled data
    return load_x
