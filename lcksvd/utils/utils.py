# -*- coding: utf-8 -*-
""" utils/utils """

from functools import wraps
from time import time

import numpy as np
import scipy.linalg as splin


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


def normcols(matrix):
    """
    Returns an array with columns normalised

    Args:
        matrix: numpy array

    Returns:
        numpy_array
    """
    return matrix/splin.norm(matrix, axis=0)


def timing(function):
    """
    Times execution time in seconds and prints it
    Args:
        function: Function or method to be timed

    Note: The first argument of the function or method must contain an attribute
          timeit = False to disable the time trancking and printing.
    """
    @wraps(function)
    def wrap(*args, **kw):
        if getattr(args[0], 'timeit', False):
            start = time()

        result = function(*args, **kw)

        if getattr(args[0], 'timeit', False):
            end = time()
            name = getattr(function, 'py_func.__qualname__', getattr(function, '__name__', function.__str__()))
            print('func:{} processed in {:.4f} seconds'.format(
                name, end-start))

        return result

    return wrap
