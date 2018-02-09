import os
import pandas as pd
import scipy.io as scio
import numpy as np
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import scoreatpercentile
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import LocalOutlierFactor

from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from utility import get_top_n, get_precn, print_baseline
from knn import Knn
from PyNomaly import loop

# load data file
# mat = scio.loadmat(os.path.join('datasets', 'arrhythmia.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'cardio.mat'))
mat = scio.loadmat(os.path.join('datasets', 'letter.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'speech.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'mammography.mat'))

ite = 30  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%

X_orig = mat['X']
y_orig = mat['y']

# outlier percentage
out_perc = np.count_nonzero(y_orig) / len(y_orig)

# initialize the score
result_dict = {}

clf_list = [XGBClassifier(), LogisticRegression(penalty="l1"),
            LogisticRegression(penalty="l2")]
clf_name_list = ['xgb', 'lr1', 'lr2']

# initialize the result dictionary
for clf_name in clf_name_list:
    result_dict[clf_name + 'roc' + 'o'] = []
    result_dict[clf_name + 'roc' + 's'] = []
    result_dict[clf_name + 'roc' + 'n'] = []

    result_dict[clf_name + 'precn' + 'o'] = []
    result_dict[clf_name + 'precn' + 's'] = []
    result_dict[clf_name + 'precn' + 'n'] = []

for t in range(ite):

    print('\nProcessing round', t, 'out of', ite)
    # split X and y for training and validation
    X, X_test, y, y_test = train_test_split(X_orig, y_orig,
                                            test_size=test_size)

    # reserve the normalized data
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    X_test_norm = scaler.transform(X_test)

    feature_list = []

    # predefined range of K
    # trim the list
    k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 150, 200, 250]

    k_list = [k for k in k_list_pre if k < X.shape[0]]

    ###########################################################################
    train_knn = np.zeros([X.shape[0], len(k_list)])
    test_knn = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn = []
    prec_knn = []

    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='largest')
        clf.fit(X)
        train_score = clf.decision_scores()
        pred_score, _ = clf.sample_scores(X_test)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                          pren=prec_n))

        feature_list.append('knn_' + str(k))
        roc_knn.append(roc)
        prec_knn.append(prec_n)

        train_knn[:, i] = train_score
        test_knn[:, i] = pred_score.ravel()
    ###########################################################################

    train_knn_mean = np.zeros([X.shape[0], len(k_list)])
    test_knn_mean = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn_mean = []
    prec_knn_mean = []
    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='mean')
        clf.fit(X)
        train_score = clf.decision_scores()
        pred_score, _ = clf.sample_scores(X_test)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn_mean roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                               pren=prec_n))

        feature_list.append('knn_mean_' + str(k))
        roc_knn_mean.append(roc)
        prec_knn_mean.append(prec_n)

        train_knn_mean[:, i] = train_score
        test_knn_mean[:, i] = pred_score.ravel()
    ###########################################################################

    train_knn_median = np.zeros([X.shape[0], len(k_list)])
    test_knn_median = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn_median = []
    prec_knn_median = []
    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='median')
        clf.fit(X)
        train_score = clf.decision_scores()
        pred_score, _ = clf.sample_scores(X_test)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn_median roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                                 pren=prec_n))

        feature_list.append('knn_median_' + str(k))
        roc_knn_median.append(roc)
        prec_knn_median.append(prec_n)

        train_knn_median[:, i] = train_score
        test_knn_median[:, i] = pred_score.ravel()
    ###########################################################################

    train_lof = np.zeros([X.shape[0], len(k_list)])
    test_lof = np.zeros([X_test.shape[0], len(k_list)])

    roc_lof = []
    prec_lof = []

    for i in range(len(k_list)):
        k = k_list[i]
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit(X)

        # save the train sets
        train_score = clf.negative_outlier_factor_ * -1
        # flip the score
        pred_score = clf._decision_function(X_test) * -1

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('lof roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                          pren=prec_n))
        feature_list.append('lof_' + str(k))
        roc_lof.append(roc)
        prec_lof.append(prec_n)

        train_lof[:, i] = train_score
        test_lof[:, i] = pred_score

    ###########################################################################
    # Noted that LoOP is not really used for prediction since its high
    # computational complexity
    # However, it is included to demonstrate the effectiveness of XGBOD

    df_X = pd.DataFrame(np.concatenate([X, X_test], axis=0))

    # predefined range of K
    k_list_pre = [1, 5, 10, 20]

    train_loop = np.zeros([X.shape[0], len(k_list_pre)])
    test_loop = np.zeros([X_test.shape[0], len(k_list_pre)])

    roc_loop = []
    prec_loop = []

    for i in range(len(k_list_pre)):
        k = k_list_pre[i]
        clf = loop.LocalOutlierProbability(df_X, n_neighbors=k).fit()
        score = clf.local_outlier_probabilities.astype(float)

        # save the train sets
        train_score = score[0:X.shape[0]]
        # flip the score
        pred_score = score[X.shape[0]:]

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('loop roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                           pren=prec_n))
        feature_list.append('loop_' + str(k))
        roc_loop.append(roc)
        prec_loop.append(prec_n)

        train_loop[:, i] = train_score
        test_loop[:, i] = pred_score

    ##########################################################################
    nu_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    train_svm = np.zeros([X.shape[0], len(nu_list)])
    test_svm = np.zeros([X_test.shape[0], len(nu_list)])

    roc_svm = []
    prec_svm = []

    for i in range(len(nu_list)):
        nu = nu_list[i]

        clf = OneClassSVM(nu=nu)
        clf.fit(X)

        train_score = clf.decision_function(X) * -1
        pred_score = clf.decision_function(X_test) * -1

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('svm roc / pren @ {nu} is {roc} {pren}'.format(nu=nu, roc=roc,
                                                             pren=prec_n))

        feature_list.append('svm_' + str(nu))
        roc_svm.append(roc)
        prec_svm.append(prec_n)

        train_svm[:, i] = train_score.ravel()
        test_svm[:, i] = pred_score.ravel()
    ###########################################################################

    n_list = [10, 20, 50, 70, 100, 150, 200, 250]

    train_if = np.zeros([X.shape[0], len(n_list)])
    test_if = np.zeros([X_test.shape[0], len(n_list)])

    roc_if = []
    prec_if = []

    for i in range(len(n_list)):
        n = n_list[i]
        clf = IsolationForest(n_estimators=n)
        clf.fit(X)
        train_score = clf.decision_function(X)
        pred_score = clf.decision_function(X_test) * -1

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('if roc / pren @ {n} is {roc} {pren}'.format(n=n, roc=roc,
                                                           pren=prec_n))

        feature_list.append('if_' + str(n))
        roc_if.append(roc)
        prec_if.append(prec_n)

        train_if[:, i] = train_score
        test_if[:, i] = pred_score

    #########################################################################
    X_train_new = np.concatenate((train_knn, train_knn_mean, train_knn_median,
                                  train_lof, train_loop, train_svm, train_if),
                                 axis=1)
    X_test_new = np.concatenate((test_knn, test_knn_mean, test_knn_median,
                                 test_lof, test_loop, test_svm, test_if),
                                axis=1)

    X_train_all = np.concatenate((X, X_train_new), axis=1)
    X_test_all = np.concatenate((X_test, X_test_new), axis=1)

    roc_list = roc_knn + roc_knn_mean + roc_knn_median + roc_lof + roc_loop + roc_svm + roc_if
    prec_n_list = prec_knn + prec_knn_mean + prec_knn_median + prec_lof + prec_loop + prec_svm + prec_if

    # get the results of baselines
    print_baseline(X_test_new, y_test, roc_list, prec_n_list)

    ##############################################################################
    for clf, clf_name in zip(clf_list, clf_name_list):
        print('processing', clf_name)
        if clf_name != 'xgb':
            clf = BalancedBaggingClassifier(base_estimator=clf,
                                            ratio='auto',
                                            replacement=False)
        # fully supervised
        clf.fit(X, y.ravel())
        y_pred = clf.predict_proba(X_test)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'roc' + 'o'].append(roc_score)
        result_dict[clf_name + 'precn' + 'o'].append(prec_n)

        # unsupervised
        clf.fit(X_train_new, y.ravel())
        y_pred = clf.predict_proba(X_test_new)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'roc' + 'n'].append(roc_score)
        result_dict[clf_name + 'precn' + 'n'].append(prec_n)

        # semi-supervised
        clf.fit(X_train_all, y.ravel())
        y_pred = clf.predict_proba(X_test_all)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'roc' + 's'].append(roc_score)
        result_dict[clf_name + 'precn' + 's'].append(prec_n)

for eva in ['roc', 'precn']:
    print()
    for clf_name in clf_name_list:
        print(np.round(np.mean(result_dict[clf_name + eva + 'o']), decimals=4),
              eva, clf_name, 'old')
        print(np.round(np.mean(result_dict[clf_name + eva + 'n']), decimals=4),
              eva, clf_name, 'new')
        print(np.round(np.mean(result_dict[clf_name + eva + 's']), decimals=4),
              eva, clf_name, 'all')
