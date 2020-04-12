# -*- coding: utf-8 -*-

import cupy as np
import cupy.linalg as splin
import scipy as sp

from .utils.linear_model.omp import orthogonal_mp_gram
from .utils.utils import normcols, timing  # , colnorms_squared_new


class ApproximateKSVD:
    def __init__(self, n_components, max_iter=10, tol=1e-6, transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components               : Number of dictionary elements
        max_iter                   : Maximum number of iterations
        tol                        : tolerance for error
        transform_n_nonzero_coefs  : Number of nonzero coefficients to target
        """
        self.components_ = None
        self.gamma_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[j, :] > 0

            if np.sum(I) == 0:
                continue

            D[:, j] = 0
            g = gamma[j][I]
            r = X[:, I] - D.dot(gamma[:, I])
            d = r.dot(g)
            d /= splin.norm(d)
            g = r.T.dot(d)
            D[:, j] = d
            gamma[j][I] = g.T

        return D, gamma

    def _initialize(self, X):
        # TODO: review these lines, several replacements were done
        __import__("pdb").set_trace()

        if np.min(X.shape) < self.n_components:
            D = np.random.randn(X.shape[0], self.n_components)
        else:
            # TODO: replace thisline with its cuda version
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(u, np.diag(s))
        D /= splin.norm(D, axis=0)[np.newaxis, :]
        return D

    def _transform(self, D, X):
        gram = D.T.dot(D)
        Xy = D.T.dot(X)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(gram, Xy, copy_Gram=False, copy_Xy=False, n_nonzero_coefs=n_nonzero_coefs)

    def fit(self, X, Dinit=None):
        """
        Use data to learn dictionary and activations.
        Parameters
        ----------
        X: data. (shape = [n_features, n_samples])
        Dinit: initialization of dictionary. (shape = [n_features, n_components])
        """
        if Dinit is None:
            D = self._initialize(X)
        else:
            D = Dinit / splin.norm(Dinit, axis=0)[np.newaxis, :]

        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = splin.norm(X - D.dot(gamma))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        self.gamma_ = gamma
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


class DKSVD:
    """
    Implementation of the Label consistent KSVD algorithm proposed by Zhuolin Jiang, Zhe Lin and Larry S. Davis.
    This implementation is a translation of the matlab code released by the authors on http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html.
    The code has been extended in order to use the related method called Discriminative KSVD proposed by Zhang, Qiang and Li, Baoxin.
    Original author: Adrien Lagrange (ad.lagrange@gmail.com)
    Date: 25-10-2018

    New version:
    Author: Giussepi Lopez (giussepexy@gmail.com)
    Date:
    """

    def __init__(self, **kwargs):
        """
        Sets initial attribute values
        Kwargs:
            sparsitythres    : sparsity threshold for KSVD
            sqrt_alpha       : contribution factor
            sqrt_beta        : contribution factor (only for LC-KSVD2)
            dictsize         : number of dictionary items
            iterations       : iterations for KSVD
            iterations4ini   : iterations when initializing
            tol              : tolerance when performing the approximate KSVD
            timeit           : Time and print functions
        """
        # TODO: because we're using another ksvd algorithm the sparsitythres
        # cannot be 30, try changing this number and or look for another implementation
        # similar to the one used on the matlab implementation
        self.sparsitythres = kwargs.get('sparsitythres', 15)  # 30
        self.sqrt_alpha = kwargs.get('sqrt_alpha', 4)
        self.sqrt_beta = kwargs.get('sqrt_beta', 2)
        self.dictsize = kwargs.get('dictsize', 570)
        self.iterations = kwargs.get('iterations', 50)
        self.iterations4ini = kwargs.get('iterations4ini', 20)
        self.tol = kwargs.get('tol', 1e-4)
        self.timeit = kwargs.get('timeit', False)

    @timing
    def initialization4LCKSVD(self, training_feats, H_train):
        """
        Initialization for Label consistent KSVD algorithm
        Args:
              training_feats  : training features
              H_train         : label matrix for training feature
        Returns:
              Dinit           : initialized dictionary
              Tinit           : initialized linear transform matrix
              Winit           : initialized classifier parameters
              Q               : optimal code matrix for training features
        """
        numClass = H_train.shape[0]  # number of classes (38, 1216)
        numPerClass = round(self.dictsize/float(numClass))  # initial points from each class 15
        Dinit = np.empty((training_feats.shape[0], numClass*numPerClass))  # for LC-Ksvd1 and LC-Ksvd2 (504, 570)

        # training_feats.shape (504, 1216)
        dictLabel = np.zeros((numClass, numClass*numPerClass))  # (38, 570)
        runKsvd = ApproximateKSVD(numPerClass, max_iter=self.iterations4ini, tol=self.tol,
                                  transform_n_nonzero_coefs=self.sparsitythres)

        for classid in range(numClass):
            col_ids = np.array(np.nonzero(H_train[classid, :] == 1)).ravel()
            # TODO: I think the following two lines are equivalent. Remove one
            data_ids = np.array(np.nonzero(np.sum(training_feats[:, col_ids]**2, axis=0) > 1e-6)).ravel()
            # NOTE: If need to conserve memory the following lines will compute it by blocks
            # data_ids = np.array(np.nonzero(colnorms_squared_new(training_feats[:, col_ids]) > 1e-6)).ravel()

            Dpart = training_feats[:, col_ids[np.random.choice(data_ids, numPerClass, replace=False)]]
            # normalization
            Dpart = normcols(Dpart)
            # ksvd process
            runKsvd.fit(training_feats[:, col_ids[data_ids]], Dpart)
            Dinit[:, numPerClass*classid:numPerClass*(classid+1)] = runKsvd.components_
            dictLabel[classid, numPerClass*classid:numPerClass*(classid+1)] = 1.

        # Q (label-constraints code); T: scale factor
        T = np.eye(self.dictsize)  # scale factor
        Q = np.zeros((self.dictsize, training_feats.shape[1]))  # energy matrix

        for frameid in range(training_feats.shape[1]):
            label_training = H_train[:, frameid]
            maxid1 = np.nonzero(label_training == np.max(label_training))[0][0]
            for itemid in range(Dinit.shape[1]):
                label_item = dictLabel[:, itemid]
                maxid2 = np.nonzero(label_item == np.max(label_item))[0][0]
                if maxid1 == maxid2:
                    Q[itemid, frameid] = 1

        # ksvd process
        runKsvd.fit(training_feats, Dinit=normcols(Dinit))
        Xtemp = runKsvd.gamma_

        # learning linear classifier parameters
        xxt = Xtemp.dot(Xtemp.T)
        tmp_result = splin.pinv(xxt+np.eye(*xxt.shape)).dot(Xtemp)
        Winit = tmp_result.dot(H_train.T)
        Tinit = tmp_result.dot(Q.T)

        return Dinit, Tinit.T, Winit.T, Q

    @timing
    def initialization4DKSVD(self, training_feats, labels, Dinit=None):
        """
        Initialization for Discriminative KSVD algorithm

        Inputs
              training_feats  : training features
              labels          : label matrix for training feature (numberred from 1 to nb of classes)
              Dinit           : initial guess for dictionary
        Outputs
              Dinit           : initialized dictionary
              Winit           : initialized classifier parameters
        """
        # H_train = sp.zeros((int(labels.max()), training_feats.shape[1]), dtype=float)
        # for c in range(int(labels.max())):
        #     H_train[c, labels == (c+1)] = 1.
        H_train = labels

        if Dinit is None:
            Dinit = training_feats[:, np.random.choice(training_feats.shape[1], self.dictsize, replace=False)]

        # ksvd process
        runKsvd = ApproximateKSVD(self.dictsize, max_iter=self.iterations4ini, tol=self.tol,
                                  transform_n_nonzero_coefs=self.sparsitythres)
        runKsvd.fit(training_feats, Dinit=normcols(Dinit))

        # learning linear classifier parameters
        Winit = splin.pinv(runKsvd.gamma_.dot(runKsvd.gamma_.T) +
                           np.eye(runKsvd.gamma_.shape[0])).dot(runKsvd.gamma_).dot(H_train.T)

        return Dinit, Winit.T

    @timing
    def labelconsistentksvd(self, Y, Dinit, labels, Q_train, Tinit, Winit=None):
        """
        Label consistent KSVD1 algorithm and Discriminative LC-KSVD2 implementation

        Args:
            Y       (cupy.ndarray) : training features
            Dinit   (cupy.ndarray) : initialized dictionary
            labels  (cupy.ndarray) : labels matrix for training feature (numberred from 1 to nb of classes)
            Q_train (cupy.ndarray) : optimal code matrix for training feature
            Tinit   (cupy.ndarray) : initialized transform matrix
            Winit   (cupy.ndarray) : initialized classifier parameters (None for LC-KSVD1)

        Returns:
            D       (cupy.ndarray) : learned dictionary
            X       (cupy.ndarray) : sparsed codes
            T       (cupy.ndarray) : learned transform matrix
            W       (cupy.ndarray) : learned classifier parameters
        """
        assert isinstance(Y,  np.ndarray)
        assert isinstance(Dinit,  np.ndarray)
        assert isinstance(labels,  np.ndarray)
        assert isinstance(Q_train,  np.ndarray)
        assert isinstance(Tinit,  np.ndarray)
        assert Winit is None or isinstance(Winit, np.ndarray)

        # H_train = sp.zeros((int(labels.max()), Y.shape[1]), dtype=float)
        # print(H_train.shape)
        # for c in range(int(labels.max())):
        #     H_train[c, labels == (c+1)] = 1.
        H_train = labels

        # ksvd process
        runKsvd = ApproximateKSVD(Dinit.shape[1], max_iter=self.iterations,
                                  tol=self.tol, transform_n_nonzero_coefs=self.sparsitythres)
        if Winit is None:
            runKsvd.fit(
                np.vstack((Y, self.sqrt_alpha*Q_train)),
                Dinit=normcols(np.vstack((Dinit, self.sqrt_alpha*Tinit)))
            )
        else:
            runKsvd.fit(
                np.vstack((Y, self.sqrt_alpha*Q_train, self.sqrt_beta*H_train)),
                Dinit=normcols(np.vstack((Dinit, self.sqrt_alpha*Tinit, self.sqrt_beta*Winit)))
            )

        # get back the desired D, T and W (if sqrt_beta is not None)
        i_end_D = Dinit.shape[0]
        i_start_T = i_end_D
        i_end_T = i_end_D+Tinit.shape[0]
        D = runKsvd.components_[:i_end_D, :]
        T = runKsvd.components_[i_start_T:i_end_T, :]
        if Winit is not None:
            i_start_W = i_end_T
            i_end_W = i_end_T+Winit.shape[0]
            W = runKsvd.components_[i_start_W:i_end_W, :]

        # normalization
        l2norms = splin.norm(D, axis=0)[np.newaxis, :] + self.tol
        D /= l2norms
        T /= l2norms
        T /= self.sqrt_alpha
        X = runKsvd.gamma_

        if Winit is None:
            # Learning linear classifier parameters
            xxt = X.dot(X.T)
            W = splin.pinv(xxt + np.eye(*(xxt).shape)).dot(X).dot(H_train.T)
            # CUSOLVERError: CUSOLVER_STATUS_EXECUTION_FAILED
            W = W.T
        else:
            W /= l2norms
            W /= self.sqrt_beta

        return D, X, T, W

    def labelconsistentksvd1(self, Y, Dinit, labels, Q_train, Tinit):
        """
        Label consistent KSVD1 algorithm
        Args:
            Y               : training features
            Dinit           : initialized dictionary
            labels          : labels matrix for training feature (numberred from 1 to nb of classes)
            Q_train         : optimal code matrix for training feature
            Tinit           : initialized transform matrix
        Returns:
            D               : learned dictionary
            X               : sparsed codes
            T               : learned transform matrix
            W               : learned classifier parameters
        """
        return self.labelconsistentksvd(Y, Dinit, labels, Q_train, Tinit)

    def labelconsistentksvd2(self, Y, Dinit, labels, Q_train, Tinit, Winit=None):
        """
        Discriminative LC-KSVD2 implementation
        Args:
            Y               : training features
            Dinit           : initialized dictionary
            labels          : labels matrix for training feature (numberred from 1 to nb of classes)
            Q_train         : optimal code matrix for training feature
            Tinit           : initialized transform matrix
            Winit           : initialized classifier parameters
        Returns:
            D               : learned dictionary
            X               : sparsed codes
            T               : learned transform matrix
            W               : learned classifier parameters
        """
        return self.labelconsistentksvd(Y, Dinit, labels, Q_train, Tinit, Winit)

    @timing
    def classification(self, D, W, data):
        """
        Classification
        Args:
            D                 : learned dictionary
            W                 : learned classifier parameters
            data              : testing features
        Returns:
            prediction_labels : prediction labels
            gamma             : learned representation
        """

        # sparse coding
        G = D.T.dot(D)
        gamma = orthogonal_mp_gram(G, D.T.dot(data), copy_Gram=False, copy_Xy=False, n_nonzero_coefs=self.sparsitythres)

        # classify process
        # TODO: find out why this error happens and fix it!
        # CUBLASError: CUBLAS_STATUS_INTERNAL_ERROR
        # when called 2nd time cupy.cuda.cublas.CUBLASError: CUBLAS_STATUS_296EXECUTION_FAI
        # It's fixed when using a float32 data; but we don't want that!
        # prediction = np.argmax(W.dot(gamma), axis=0)
        # Temporary workaround:
        # Performing the last dot product on CPU through numpy
        prediction = np.argmax(np.array(np.asnumpy(W).dot(np.asnumpy(gamma))), axis=0)

        return prediction, gamma
