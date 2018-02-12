import numpy as np
import pandas as pd
from models.utility import get_precn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from PyNomaly import loop
from models.hbos import Hbos


def knn(X, n_neighbors):
    '''
    Utility function to return k-average, k-median, knn
    Since these three functions are similar, so is inluded in the same func
    :param X: train data
    :param n_neighbors: number of neighbors
    :return:
    '''
    neigh = NearestNeighbors()
    neigh.fit(X)

    res = neigh.kneighbors(n_neighbors=n_neighbors, return_distance=True)
    # k-average, k-median, knn
    return np.mean(res[0], axis=1), np.median(res[0], axis=1), res[0][:, -1]


def get_TOS_knn(X, y, k_list, feature_list):
    knn_clf = ["knn_mean", "knn_median", "knn_kth"]

    result_knn = np.zeros([X.shape[0], len(k_list) * len(knn_clf)])
    roc_knn = []
    prec_knn = []

    for i in range(len(k_list)):
        k = k_list[i]
        k_mean, k_median, k_k = knn(X, n_neighbors=k)
        knn_result = [k_mean, k_median, k_k]

        for j in range(len(knn_result)):
            score_pred = knn_result[j]
            clf = knn_clf[j]

            roc = np.round(roc_auc_score(y, score_pred), decimals=4)
            # apc = np.round(average_precision_score(y, score_pred), decimals=4)
            prec_n = np.round(get_precn(y, score_pred), decimals=4)
            print('{clf} @ {k} - ROC: {roc} Precision@n: {pren}'.
                  format(clf=clf, k=k, roc=roc, pren=prec_n))
            feature_list.append(clf + str(k))
            roc_knn.append(roc)
            prec_knn.append(prec_n)
            result_knn[:, i * len(knn_result) + j] = score_pred

    print()
    return feature_list, roc_knn, prec_knn, result_knn


def get_TOS_loop(X, y, k_list, feature_list):
    # only compatible with pandas
    df_X = pd.DataFrame(X)

    result_loop = np.zeros([X.shape[0], len(k_list)])
    roc_loop = []
    prec_loop = []

    for i in range(len(k_list)):
        k = k_list[i]
        clf = loop.LocalOutlierProbability(df_X, n_neighbors=k).fit()
        score_pred = clf.local_outlier_probabilities.astype(float)

        roc = np.round(roc_auc_score(y, score_pred), decimals=4)
        # apc = np.round(average_precision_score(y, score_pred), decimals=4)
        prec_n = np.round(get_precn(y, score_pred), decimals=4)

        print('LoOP @ {k} - ROC: {roc} Precision@n: {pren}'.format(k=k,
                                                                   roc=roc,
                                                                   pren=prec_n))

        feature_list.append('loop_' + str(k))
        roc_loop.append(roc)
        prec_loop.append(prec_n)
        result_loop[:, i] = score_pred
    print()
    return feature_list, roc_loop, prec_loop, result_loop


def get_TOS_lof(X, y, k_list, feature_list):
    result_lof = np.zeros([X.shape[0], len(k_list)])
    roc_lof = []
    prec_lof = []

    for i in range(len(k_list)):
        k = k_list[i]
        clf = LocalOutlierFactor(n_neighbors=k)
        y_pred = clf.fit_predict(X)
        score_pred = clf.negative_outlier_factor_

        roc = np.round(roc_auc_score(y, score_pred * -1), decimals=4)
        # apc = np.round(average_precision_score(y, score_pred * -1), decimals=4)
        prec_n = np.round(get_precn(y, score_pred * -1), decimals=4)
        print('LOF @ {k} - ROC: {roc} Precision@n: {pren}'.format(k=k,
                                                                  roc=roc,
                                                                  pren=prec_n))

        feature_list.append('lof_' + str(k))
        roc_lof.append(roc)
        prec_lof.append(prec_n)
        result_lof[:, i] = score_pred * -1
    print()
    return feature_list, roc_lof, prec_lof, result_lof


def get_TOS_hbos(X, y, k_list, feature_list):
    result_hbos = np.zeros([X.shape[0], len(k_list)])
    roc_hbos = []
    prec_hbos = []

    k_list = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
    for i in range(len(k_list)):
        k = k_list[i]
        clf = Hbos(bins=k, alpha=0.3)
        clf.fit(X)
        score_pred = clf.decision_scores

        roc = np.round(roc_auc_score(y, score_pred), decimals=4)
        # apc = np.round(average_precision_score(y, score_pred * -1), decimals=4)
        prec_n = np.round(get_precn(y, score_pred), decimals=4)
        print('HBOS @ {k} - ROC: {roc} Precision@n: {pren}'.format(k=k,
                                                                   roc=roc,
                                                                   pren=prec_n))

        feature_list.append('hbos_' + str(k))
        roc_hbos.append(roc)
        prec_hbos.append(prec_n)
        result_hbos[:, i] = score_pred
    print()
    return feature_list, roc_hbos, prec_hbos, result_hbos


def get_TOS_svm(X, y, nu_list, feature_list):
    result_ocsvm = np.zeros([X.shape[0], len(nu_list)])
    roc_ocsvm = []
    prec_ocsvm = []

    for i in range(len(nu_list)):
        nu = nu_list[i]
        clf = OneClassSVM(nu=nu)
        clf.fit(X)
        score_pred = clf.decision_function(X)

        roc = np.round(roc_auc_score(y, score_pred * -1), decimals=4)

        # apc = np.round(average_precision_score(y, score_pred * -1), decimals=4)
        prec_n = np.round(
            get_precn(y, score_pred * -1), decimals=4)
        print('svm @ {nu} - ROC: {roc} Precision@n: {pren}'.format(nu=nu,
                                                                   roc=roc,
                                                                   pren=prec_n))
        feature_list.append('ocsvm_' + str(nu))
        roc_ocsvm.append(roc)
        prec_ocsvm.append(prec_n)
        result_ocsvm[:, i] = score_pred.reshape(score_pred.shape[0]) * -1
    print()
    return feature_list, roc_ocsvm, prec_ocsvm, result_ocsvm


def get_TOS_iforest(X, y, n_list, feature_list):
    result_if = np.zeros([X.shape[0], len(n_list)])
    roc_if = []
    prec_if = []

    for i in range(len(n_list)):
        n = n_list[i]
        clf = IsolationForest(n_estimators=n)
        clf.fit(X)
        score_pred = clf.decision_function(X)

        roc = np.round(roc_auc_score(y, score_pred * -1), decimals=4)
        prec_n = np.round(get_precn(y, y_pred=(score_pred * -1)), decimals=4)

        print('Isolation Forest @ {n} - ROC: {roc} Precision@n: {pren}'.format(
            n=n,
            roc=roc,
            pren=prec_n))
        feature_list.append('if_' + str(n))
        roc_if.append(roc)
        prec_if.append(prec_n)
        result_if[:, i] = score_pred.reshape(score_pred.shape[0]) * -1
    print()
    return feature_list, roc_if, prec_if, result_if
