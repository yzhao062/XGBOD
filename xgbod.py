import pandas as pd
import scipy as sc
import scipy.io as scio
import numpy as np
import random
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from utility import precision_n, get_top_n, print_baseline
from generate_TOS import generate_TOS_knn
from generate_TOS import generate_TOS_loop
from generate_TOS import generate_TOS_lof
from generate_TOS import generate_TOS_svm
from generate_TOS import generate_TOS_iforest

from select_TOS import random_select, accurate_select, balance_select

# load data file
# mat = scio.loadmat('cardio.mat')
mat = scio.loadmat('arrhythmia.mat')

X = mat['X']
y = mat['y']

# knn, LoOP, and LOF use normalized X
X_norm = normalize(X)
feature_list = []

# Running KNN-base algorithms to generate addtional features

# predefined range of k
k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150,
              200, 250]
# predefined range of k to be used with LoOp dur to high complexity
k_list_pre_short = [1, 3, 5, 10]

# validate the value of k
k_list = [k for k in k_list_pre if k < X.shape[0]]

# predefined range of nu for one-class svm
nu_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# predefined range for number of estimators in isolation forests
n_list = [10, 20, 50, 70, 100, 150, 200, 250]
##############################################################################
# # Generate TOS using KNN based algorithms
(feature_list, roc_knn, prec_knn, result_knn) = generate_TOS_knn(X_norm, y,
                                                                 k_list,
                                                                 feature_list)

# Generate TOS using LoOP
(feature_list, roc_loop, prec_loop, result_loop) = generate_TOS_loop(X_norm, y,
                                                                     k_list_pre_short,
                                                                     feature_list)
# Generate TOS using lof
(feature_list, roc_lof, prec_lof, result_lof) = generate_TOS_lof(X_norm, y,
                                                                 k_list_pre,
                                                                 feature_list)
# Generate TOS using one class svm
(feature_list, roc_ocsvm, prec_ocsvm, result_ocsvm) = generate_TOS_svm(X, y,
                                                                       nu_list,
                                                                       feature_list)
# Generate TOS using isolation forests
(feature_list, roc_if, prec_if, result_if) = generate_TOS_iforest(X, y,
                                                                  n_list,
                                                                  feature_list)
##############################################################################
# combine the feature space
X_train_new_orig = np.concatenate(
    (result_knn, result_loop, result_lof, result_ocsvm, result_if), axis=1)

X_train_all_orig = np.concatenate((X, X_train_new_orig), axis=1)

# combine ROC and Precision@n list
roc_list = roc_knn + roc_loop + roc_lof + roc_ocsvm + roc_if
prec_list = prec_knn + prec_loop + prec_lof + prec_ocsvm + prec_if
#

print_baseline(X_train_new_orig, y, roc_list, prec_list)

##############################################################################
# select TOS using different methods

p = 10  # number of selected TOS
# random selection
X_train_new_rand, X_train_all_rand = random_select(X, X_train_new_orig,
                                                   roc_list, p)
# print(X_train_new_rand.shape, X_train_all_rand.shape)
# accurate selection
X_train_new_accu, X_train_all_accu = accurate_select(X, X_train_new_orig,
                                                     feature_list, roc_list, p)
# print(X_train_new_accu.shape, X_train_all_accu.shape)

# balance selection
X_train_new_bal, X_train_all_bal = balance_select(X, X_train_new_orig,
                                                  roc_list, p)
# print(X_train_new_bal.shape, X_train_all_bal.shape)

###############################################################################
# build various classifiers

ite = 30  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%
result_dict = {}

clf_list = [XGBClassifier(), LogisticRegression(penalty="l1"),
            LogisticRegression(penalty="l2")]
clf_name_list = ['xgb', 'lr1', 'lr2']

# initialize the result dictionary
for clf_name in clf_name_list:
    result_dict[clf_name + 'roc' + 'o'] = []
    result_dict[clf_name + 'roc' + 's'] = []
    result_dict[clf_name + 'roc' + 'n'] = []

    result_dict[clf_name + 'prec' + 'o'] = []
    result_dict[clf_name + 'prec' + 's'] = []
    result_dict[clf_name + 'prec' + 'n'] = []

    result_dict[clf_name + 'precn' + 'o'] = []
    result_dict[clf_name + 'precn' + 's'] = []
    result_dict[clf_name + 'precn' + 'n'] = []

for i in range(ite):

    s_feature_rand = random.sample(range(0, len(roc_list)), p)
    X_train_new_rand = X_train_new_orig[:, s_feature_rand]
    X_train_all_rand = np.concatenate((X, X_train_new_rand), axis=1)

    original_len = X.shape[1]

    # use all TOS
    X_train, X_test, y_train, y_test = train_test_split(X_train_all_orig, y,
                                                        test_size=test_size)
    # # use Random Selection
    # X_train, X_test, y_train, y_test = train_test_split(X_train_all_rand, y,
    #                                                     test_size=test_size)
    # # use Accurate Selection
    # X_train, X_test, y_train, y_test = train_test_split(X_train_all_accu, y,
    #                                                     test_size=test_size)
    # # use Balance Selection
    # X_train, X_test, y_train, y_test = train_test_split(X_train_all_bal, y,
    #                                                     test_size=test_size)

    # use original features
    X_train_o = X_train[:, 0:original_len]
    X_test_o = X_test[:, 0:original_len]

    X_train_n = X_train[:, original_len:]
    X_test_n = X_test[:, original_len:]

    for clf, clf_name in zip(clf_list, clf_name_list):

        print('processing', clf_name, 'round', i + 1)
        if clf_name != 'xgb':
            clf = BalancedBaggingClassifier(base_estimator=clf,
                                            ratio='auto',
                                            replacement=False)

        # fully supervised
        clf.fit(X_train_o, y_train.ravel())
        y_pred = clf.predict_proba(X_test_o)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        avg_prec = average_precision_score(y_test, y_pred[:, 1])
        prec_n = precision_n(y=y_test.ravel(), y_pred=y_pred[:, 1],
                             n=y_test.sum())

        result_dict[clf_name + 'roc' + 'o'].append(roc_score)
        result_dict[clf_name + 'prec' + 'o'].append(avg_prec)
        result_dict[clf_name + 'precn' + 'o'].append(prec_n)

        # unsupervised
        clf.fit(X_train_n, y_train.ravel())
        y_pred = clf.predict_proba(X_test_n)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        avg_prec = average_precision_score(y_test, y_pred[:, 1])
        prec_n = precision_n(y=y_test.ravel(), y_pred=y_pred[:, 1],
                             n=y_test.sum())

        result_dict[clf_name + 'roc' + 'n'].append(roc_score)
        result_dict[clf_name + 'prec' + 'n'].append(avg_prec)
        result_dict[clf_name + 'precn' + 'n'].append(avg_prec)

        # semi-supervised
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict_proba(X_test)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        avg_prec = average_precision_score(y_test, y_pred[:, 1])
        prec_n = precision_n(y=y_test.ravel(), y_pred=y_pred[:, 1],
                             n=y_test.sum())

        result_dict[clf_name + 'roc' + 's'].append(roc_score)
        result_dict[clf_name + 'prec' + 's'].append(avg_prec)
        result_dict[clf_name + 'precn' + 's'].append(avg_prec)

for eva in ['roc', 'precn']:
    print()
    for clf_name in clf_name_list:
        print(np.round(np.mean(result_dict[clf_name + eva + 'o']), decimals=4),
              eva, clf_name, 'original features')
        print(np.round(np.mean(result_dict[clf_name + eva + 'n']), decimals=4),
              eva, clf_name, 'TOS only')
        print(np.round(np.mean(result_dict[clf_name + eva + 's']), decimals=4),
              eva, clf_name, 'original feature + TOS')
