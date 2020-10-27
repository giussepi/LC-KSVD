# -*- coding: utf-8 -*-
""" main """

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from lcksvd.dksvd import DKSVD


def main():

    ###########################################################################
    #                                 LC-KSVD1                                #
    ###########################################################################
    # file_path = 'trainingdata/featurevectors.mat'
    # data = loadmat(file_path)
    # lcksvd = DKSVD(timeit=True)
    # Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(data['training_feats'], data['H_train'])
    # D, X, T, W = lcksvd.labelconsistentksvd1(data['training_feats'], Dinit, data['H_train'], Q, Tinit_T)
    # predictions, gamma = lcksvd.classification(D, W, data['testing_feats'])
    # print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
    # # Final recognition rate for LC-KSVD1 is : 0.9215

    ###########################################################################
    #                                 LC-KSVD2                                #
    ###########################################################################
    file_path = 'trainingdata/featurevectors.mat'
    data = loadmat(file_path)
    lcksvd = DKSVD(timeit=True)
    Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(data['training_feats'], data['H_train'])

    D, X, T, W = lcksvd.labelconsistentksvd2(data['training_feats'], Dinit, data['H_train'], Q, Tinit_T, Winit_T)
    predictions, gamma = lcksvd.classification(D, W, data['testing_feats'])
    print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
        accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
    # Final recognition rate for LC-KSVD2 is : 0.9073

    ###########################################################################
    #                                  D-KSVD                                  #
    ###########################################################################
    # file_path = 'trainingdata/featurevectors.mat'
    # data = loadmat(file_path)
    # lcksvd = DKSVD(timeit=True)
    # Dinit, Winit = lcksvd.initialization4DKSVD(data['training_feats'], data['H_train'])
    # predictions, gamma = lcksvd.classification(Dinit, Winit, data['testing_feats'])
    # print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
    # # Final recognition rate for D-KSVD is : 0.8581


if __name__ == '__main__':
    main()
