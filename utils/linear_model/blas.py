# -*- coding: utf-8 -*-
""" utils/linear_model/blas """

import functools

import cupy as _np
from scipy.linalg.blas import _blas_alias, _type_score, _type_conv,\
    _fblas, _cblas


def _memoize_get_funcs(func):
    """
    Memoized fast path for _get_funcs instances
    """
    memo = {}
    func.memo = memo

    @functools.wraps(func)
    def getter(names, arrays=(), dtype=None):
        key = (names, dtype)
        for array in arrays:
            # c.f. find_blas_funcs
            # NOTE: CuPy does not have .flags.fortran attribute
            # key += (array.dtype.char, array.flags.fortran)
            key += (array.dtype.char, array.flags['F_CONTIGUOUS'])

        try:
            value = memo.get(key)
        except TypeError:
            # unhashable key etc.
            key = None
            value = None

        if value is not None:
            return value

        value = func(names, arrays, dtype)

        if key is not None:
            memo[key] = value

        return value

    return getter


def find_best_blas_type(arrays=(), dtype=None):
    """
    Based on: scipy.linalg.blas.find_best_blas_type

    Find best-matching BLAS/LAPACK type.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    prefix : str
        BLAS/LAPACK prefix character.
    dtype : dtype
        Inferred Numpy data type.
    prefer_fortran : bool
        Whether to prefer Fortran order routines over C order.

    Examples
    --------
    >>> import scipy.linalg.blas as bla
    >>> a = np.random.rand(10,15)
    >>> b = np.asfortranarray(a)  # Change the memory layout order
    >>> bla.find_best_blas_type((a,))
    ('d', dtype('float64'), False)
    >>> bla.find_best_blas_type((a*1j,))
    ('z', dtype('complex128'), False)
    >>> bla.find_best_blas_type((b,))
    ('d', dtype('float64'), True)

    """
    dtype = _np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False

    if arrays:
        # In most cases, single element is passed through, quicker route
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            # NOTE: CuPy does not have .flags.fortran attribute
            # prefer_fortran = arrays[0].flags['FORTRAN']
            prefer_fortran = arrays[0].flags['F_CONTIGUOUS']
        else:
            # use the most generic type in arrays
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            # safe upcasting for mix of float64 and complex64 --> prefix 'z'
            if max_score == 3 and (2 in scores):
                max_score = 4

            # NOTE: CuPy does not have .flags.fortran attribute
            # if arrays[ind_max_score].flags['FORTRAN']:
            if arrays[ind_max_score].flags['F_CONTIGUOUS']:
                # prefer Fortran for leading array with column major order
                prefer_fortran = True

    # Get the LAPACK prefix and the corresponding dtype if not fall back
    # to 'd' and double precision float.
    prefix, dtype = _type_conv.get(max_score, ('d', _np.dtype('float64')))

    return prefix, dtype, prefer_fortran


def _get_funcs(names, arrays, dtype,
               lib_name, fmodule, cmodule,
               fmodule_name, cmodule_name, alias):
    """
    Based on: scipy.linalg.blas._get_funcs

    Return available BLAS/LAPACK functions.

    Used also in lapack.py. See get_blas_funcs for docstring.
    """

    funcs = []
    unpack = False
    dtype = _np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)

    if isinstance(names, str):
        names = (names,)
        unpack = True

    prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)

    if prefer_fortran:
        module1, module2 = module2, module1

    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(
                '%s function %s could not be found' % (lib_name, func_name))
        func.module_name, func.typecode = module_name, prefix
        func.dtype = dtype
        func.prefix = prefix  # Backward compatibility
        funcs.append(func)

    if unpack:
        return funcs[0]

    return funcs


@_memoize_get_funcs
def get_blas_funcs(names, arrays=(), dtype=None):
    """
    Based on: scipy.linalg.blas.get_blas_funcs

    Return available BLAS function objects from names.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.


    Returns
    -------
    funcs : list
        List containing the found function(s).


    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
    types {float32, float64, complex64, complex128} respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.

    Examples
    --------
    >>> import scipy.linalg as LA
    >>> a = np.random.rand(3,2)
    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))
    >>> x_gemv.typecode
    'd'
    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
    >>> x_gemv.typecode
    'z'

    """
    return _get_funcs(names, arrays, dtype,
                      "BLAS", _fblas, _cblas, "fblas", "cblas",
                      _blas_alias)
