# -*- coding: utf-8 -*-
""" utils/linear_model/omp """

from math import sqrt
import warnings

import cupy as np
from cupyx.scipy import linalg
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda import cusolver, cublas
from sklearn.linear_model._omp import premature

from ..validation import check_array


def _gram_omp(Gram, Xy, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    """
    NOTE: Based on sklearn.linear_model._omp._gram_omp

    Orthogonal Matching Pursuit step on a precomputed Gram matrix.

    This function uses the Cholesky decomposition method.

    Parameters
    ----------
    Gram : array, shape (n_features, n_features)
        Gram matrix of the input data matrix

    Xy : array, shape (n_features,)
        Input targets

    n_nonzero_coefs : int
        Targeted number of non-zero elements

    tol_0 : float
        Squared norm of y, required if tol is not None.

    tol : float
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_Gram : bool, optional
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, optional
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, optional. Default: False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : array, shape (n_nonzero_coefs,)
        Non-zero elements of the solution

    idx : array, shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector

    coefs : array, shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    # NOTE: CUPY does not have Xy.flags.writeable
    # if copy_Xy or not Xy.flags.writeable:
    if copy_Xy:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)
    L[0, 0] = 1.

    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    check_finite=False)
            v = np.power(np.linalg.norm(L[n_active, :n_active]), 2)
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:  # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = np.sqrt(Gram[lam, lam])

        Gram[[n_active, lam]] = Gram[[lam, n_active]]
        Gram.T[[n_active, lam]] = Gram.T[[lam, n_active]]
        # NOTE: These 2 lines looks wrong but seems that this is how it has been done
        # on sklearn.
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        cuda_solver_dn_handle = cusolver.cusolverDnCreate()
        # NOTE: If there are problems because the GPU does not support float64/double
        # then just change np.float64 and cusolver.cusolverDnDpotrs to np.float32 and
        # cusolver.cusolverDnSpotrs respectively
        # change here
        A = gpuarray.to_gpu(np.asnumpy(L[:n_active, :n_active].copy(), order='F')).astype(np.float64)
        # stream = cusolver.cusolverDnGetStream(cuda_solver_dn_handle)
        # cusolver.cusolverDnSetStream(cuda_solver_dn_handle, stream)
        uplo = cublas._CUBLAS_FILL_MODE['l']
        # lwork = cusolver.cusolverDnDpotrf_bufferSize(
        #     handle=cuda_solver_dn_handle, uplo=uplo, n=L[:n_active, :n_active].shape[0],
        #     a=A.gpudata, lda=L[:n_active, :n_active].shape[0]
        # )
        # workspace_gpu = gpuarray.zeros(lwork, dtype=A.dtype)
        # dev_info_gpu = gpuarray.zeros(1, np_.int32)
        # cusolver.cusolverDnDpotrf(
        #     handle=cuda_solver_dn_handle, uplo=uplo, n=L[:n_active, :n_active].shape[0],
        #     a=A.gpudata, lda=L[:n_active, :n_active].shape[0],
        #     workspace=workspace_gpu.gpudata, devIpiv=lwork, devInfo=dev_info_gpu.gpudata
        # )
        dev_info_gpu_2 = gpuarray.zeros(1, np.int32)
        # change here
        solution = gpuarray.to_gpu(np.asnumpy(Xy[:n_active])).astype(np.float64)
        # change here
        cusolver.cusolverDnDpotrs(
            handle=cuda_solver_dn_handle, uplo=uplo, n=L[:n_active, :n_active].shape[0],
            nrhs=1, a=A.gpudata,
            lda=L[:n_active, :n_active].shape[0],
            B=solution.gpudata, ldb=solution.shape[0], devInfo=dev_info_gpu_2.gpudata
        )
        cusolver.cusolverDnDestroy(cuda_solver_dn_handle)
        gamma = np.array(solution.get())

        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        beta = np.dot(Gram[:, :n_active], gamma)
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            if abs(tol_curr) <= tol:
                break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active

    return gamma, indices[:n_active], n_active


def orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=None, tol=None,
                       norms_squared=None, copy_Gram=True,
                       copy_Xy=True, return_path=False,
                       return_n_iter=False):
    """
    NOTE: Based on sklearn.linear_model._omp.orthogonal_mp_gram

    Gram Orthogonal Matching Pursuit (OMP)

    Solves n_targets Orthogonal Matching Pursuit problems using only
    the Gram matrix X.T * X and the product X.T * y.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    Gram : array, shape (n_features, n_features)
        Gram matrix of the input data: X.T * X

    Xy : array, shape (n_features,) or (n_features, n_targets)
        Input targets multiplied by X: X.T * y

    n_nonzero_coefs : int
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float
        Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

    norms_squared : array-like, shape (n_targets,)
        Squared L2 norms of the lines of y. Required if tol is not None.

    copy_Gram : bool, optional
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, optional
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, optional. Default: False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, optional default False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : array, shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        (n_features, n_features) or (n_features, n_targets, n_features) and
        iterating over the last axis yields coefficients in increasing order
        of active features.

    n_iters : array-like or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See also
    --------
    OrthogonalMatchingPursuit
    orthogonal_mp
    lars_path
    decomposition.sparse_encode

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    """
    Gram = check_array(Gram, order='F', copy=copy_Gram)
    Xy = np.asarray(Xy)
    if Xy.ndim > 1 and Xy.shape[1] > 1:
        # or subsequent target will be affected
        copy_Gram = True
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
        if tol is not None:
            norms_squared = [norms_squared]
    # NOTE: CUPY does not have Xy.flags.writeable
    # if copy_Xy or not Xy.flags.writeable:
    if copy_Xy:
        # Make the copy once instead of many times in _gram_omp itself.
        Xy = Xy.copy()
    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = int(0.1 * len(Gram))
    if tol is not None and norms_squared is None:
        raise ValueError('Gram OMP needs the precomputed norms in order '
                         'to evaluate the error sum of squares.')
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > len(Gram):
        raise ValueError("The number of atoms cannot be more than the number "
                         "of features")

    if return_path:
        coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)))
    else:
        coef = np.zeros((len(Gram), Xy.shape[1]))

    n_iters = []
    for k in range(Xy.shape[1]):
        out = _gram_omp(
            Gram, Xy[:, k], n_nonzero_coefs,
            norms_squared[k] if tol is not None else None, tol,
            copy_Gram=copy_Gram, copy_Xy=False,
            return_path=return_path)
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, :len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[:n_active + 1], k, n_active] = x[:n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if Xy.shape[1] == 1:
        n_iters = n_iters[0]

    if return_n_iter:
        return np.squeeze(coef), n_iters

    return np.squeeze(coef)
