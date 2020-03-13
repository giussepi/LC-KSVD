# -*- coding: utf-8 -*-
""" utils/utils """

import numpy as np


def colnorms_squared_new(x):
    """
    Calculate and returns the norms of the columns.
    Note: Compute in blocks to conserve memory

    Args:
        x: numpy array

    Returns:
        numpy_array

    """
    y = np.zeros(x.shape[1])
    blocksize = 2000
    for i in range(0, x.shape[1], blocksize):
        blockids = list(range(min(i+blocksize-1, x.shape[1])))
        y[blockids] = sum(x[:, blockids]**2)

    return y
