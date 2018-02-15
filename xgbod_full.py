import os
import pandas as pd
import numpy as np
import scipy.io as scio

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from PyNomaly import loop

from models.knn import Knn
from models.utility import get_precn, print_baseline

# use one dataset at a time; more datasets could be added to /datasets folder
# the experiment codes use a bit more setting up, otherwise the
# exact reproduction is infeasible. Clean-up codes are going to be moved

# load data file
mat = scio.loadmat(os.path.join('datasets', 'letter.mat'))
ite = 30  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%

X_orig = mat['X']
y_orig = mat['y']

# outlier percentage
out_perc = np.count_nonzero(y_orig) / len(y_orig)

# define classifiers to use
clf_list = [XGBClassifier(), LogisticRegression(penalty="l1"),
            LogisticRegression(penalty="l2")]
clf_name_list = ['xgb', 'lr1', 'lr2']

# initialize the container to store the results
result_dict = {}

# initialize the result dictionary
for clf_name in clf_name_list:
    result_dict[clf_name + 'roc' + 'o'] = []
    result_dict[clf_name + 'roc' + 's'] = []
    result_dict[clf_name + 'roc' + 'n'] = []

    result_dict[clf_name + 'precn' + 'o'] = []
    result_dict[clf_name + 'precn' + 's'] = []
    result_dict[clf_name + 'precn' + 'n'] = []

for t in range(ite):

    print('\nProcessing trial', t + 1, 'out of', ite)

    # split X and y for training and validation
    X, X_test, y, y_test = train_test_split(X_orig, y_orig,
                                            test_size=test_size)

    # reserve the normalized data
    scaler = Normalizer().fit(X)
    X_norm = scaler.transform(X)
    X_test_norm = scaler.transform(X_test)

    feature_list = []

    # predefined range of K
    k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 150, 200, 250]
    # trim the list in case of small sample size
    k_list = [k for k in k_list_pre if k < X.shape[0]]

    ###########################################################################
    train_knn = np.zeros([X.shape[0], len(k_list)])
    test_knn = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn = []
    prec_n_knn = []

    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='largest')
        clf.fit(X_norm)
        train_score = clf.decision_scores
        pred_score = clf.decision_function(X_test_norm)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                          pren=prec_n))

        feature_list.append('knn_' + str(k))
        roc_knn.append(roc)
        prec_n_knn.append(prec_n)

        train_knn[:, i] = train_score
        test_knn[:, i] = pred_score.ravel()
    ###########################################################################

    train_knn_mean = np.zeros([X.shape[0], len(k_list)])
    test_knn_mean = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn_mean = []
    prec_n_knn_mean = []
    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='mean')
        clf.fit(X_norm)
        train_score = clf.decision_scores
        pred_score = clf.decision_function(X_test_norm)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn_mean roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                               pren=prec_n))

        feature_list.append('knn_mean_' + str(k))
        roc_knn_mean.append(roc)
        prec_n_knn_mean.append(prec_n)

        train_knn_mean[:, i] = train_score
        test_knn_mean[:, i] = pred_score.ravel()
    ###########################################################################

    train_knn_median = np.zeros([X.shape[0], len(k_list)])
    test_knn_median = np.zeros([X_test.shape[0], len(k_list)])

    roc_knn_median = []
    prec_n_knn_median = []
    for i in range(len(k_list)):
        k = k_list[i]

        clf = Knn(n_neighbors=k, contamination=out_perc, method='median')
        clf.fit(X_norm)
        train_score = clf.decision_scores
        pred_score = clf.decision_function(X_test_norm)

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('knn_median roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                                 pren=prec_n))

        feature_list.append('knn_median_' + str(k))
        roc_knn_median.append(roc)
        prec_n_knn_median.append(prec_n)

        train_knn_median[:, i] = train_score
        test_knn_median[:, i] = pred_score.ravel()
    ###########################################################################

    train_lof = np.zeros([X.shape[0], len(k_list)])
    test_lof = np.zeros([X_test.shape[0], len(k_list)])

    roc_lof = []
    prec_n_lof = []

    for i in range(len(k_list)):
        k = k_list[i]
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit(X_norm)

        # save the train sets
        train_score = clf.negative_outlier_factor_ * -1
        # flip the score
        pred_score = clf._decision_function(X_test_norm) * -1

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('lof roc pren @ {k} is {roc} {pren}'.format(k=k, roc=roc,
                                                          pren=prec_n))
        feature_list.append('lof_' + str(k))
        roc_lof.append(roc)
        prec_n_lof.append(prec_n)

        train_lof[:, i] = train_score
        test_lof[:, i] = pred_score

    ###########################################################################
    # Noted that LoOP is not really used for prediction since its high
    # computational complexity
    # However, it is included to demonstrate the effectiveness of XGBOD only

    df_X = pd.DataFrame(np.concatenate([X_norm, X_test_norm], axis=0))

    # predefined range of K
    k_list = [1, 5, 10, 20]

    train_loop = np.zeros([X.shape[0], len(k_list)])
    test_loop = np.zeros([X_test.shape[0], len(k_list)])

    roc_loop = []
    prec_n_loop = []

    for i in range(len(k_list)):
        k = k_list[i]
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
        prec_n_loop.append(prec_n)

        train_loop[:, i] = train_score
        test_loop[:, i] = pred_score

    ##########################################################################
    nu_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    train_svm = np.zeros([X.shape[0], len(nu_list)])
    test_svm = np.zeros([X_test.shape[0], len(nu_list)])

    roc_svm = []
    prec_n_svm = []

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
        prec_n_svm.append(prec_n)

        train_svm[:, i] = train_score.ravel()
        test_svm[:, i] = pred_score.ravel()
    ###########################################################################

    n_list = [10, 20, 50, 70, 100, 150, 200, 250]

    train_if = np.zeros([X.shape[0], len(n_list)])
    test_if = np.zeros([X_test.shape[0], len(n_list)])

    roc_if = []
    prec_n_if = []

    for i in range(len(n_list)):
        n = n_list[i]
        clf = IsolationForest(n_estimators=n)
        clf.fit(X)
        train_score = clf.decision_function(X) * -1
        pred_score = clf.decision_function(X_test) * -1

        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)
        print('if roc / pren @ {n} is {roc} {pren}'.format(n=n, roc=roc,
                                                           pren=prec_n))

        feature_list.append('if_' + str(n))
        roc_if.append(roc)
        prec_n_if.append(prec_n)

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
    prec_n_list = prec_n_knn + prec_n_knn_mean + prec_n_knn_median + prec_n_lof + prec_n_loop + prec_n_svm + prec_n_if

    # get the results of baselines
    print_baseline(X_test_new, y_test, roc_list, prec_n_list)

    ###########################################################################
    # select TOS using different methods

    p = 10  # number of selected TOS
    # TODO: supplement the cleaned up version for selection methods

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
