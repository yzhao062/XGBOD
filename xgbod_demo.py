'''
Demo codes for XGBOD.
Author: Yue Zhao

notes: the demo code simulates the use of XGBOD with some changes to expedite
the execution. Use the full code for the production.

'''
import os
import random
import scipy.io as scio
import numpy as np

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from models.utility import get_precn, print_baseline
from models.generate_TOS import get_TOS_knn
from models.generate_TOS import get_TOS_loop
from models.generate_TOS import get_TOS_lof
from models.generate_TOS import get_TOS_svm
from models.generate_TOS import get_TOS_iforest
from models.generate_TOS import get_TOS_hbos
from models.select_TOS import random_select, accurate_select, balance_select

# load data file
# mat = scio.loadmat(os.path.join('datasets', 'speech.mat'))
mat = scio.loadmat(os.path.join('datasets', 'arrhythmia.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'cardio.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'letter.mat'))
# mat = scio.loadmat(os.path.join('datasets', 'mammography.mat'))

X = mat['X']
y = mat['y']

# use unit norm vector X improves knn, LoOP, and LOF results
scaler = StandardScaler().fit(X)
# X_norm = scaler.transform(X)
X_norm = normalize(X)
feature_list = []

# Running KNN-base algorithms to generate addtional features

# predefined range of k
k_range = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150,
           200, 250]
# predefined range of k to be used with LoOP due to high complexity
k_range_short = [1, 3, 5, 10]

# validate the value of k
k_range = [k for k in k_range if k < X.shape[0]]

# predefined range of nu for one-class svm
nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# predefined range for number of estimators in isolation forests
n_range = [10, 20, 50, 70, 100, 150, 200, 250]
##############################################################################

# Generate TOS using KNN based algorithms
feature_list, roc_knn, prc_n_knn, result_knn = get_TOS_knn(X_norm, y, k_range,
                                                           feature_list)
# Generate TOS using LoOP
feature_list, roc_loop, prc_n_loop, result_loop = get_TOS_loop(X, y,
                                                               k_range_short,
                                                               feature_list)
# Generate TOS using LOF
feature_list, roc_lof, prc_n_lof, result_lof = get_TOS_lof(X_norm, y, k_range,
                                                           feature_list)
# Generate TOS using one class svm
feature_list, roc_ocsvm, prc_n_ocsvm, result_ocsvm = get_TOS_svm(X, y,
                                                                 nu_range,
                                                                 feature_list)
# Generate TOS using isolation forests
feature_list, roc_if, prc_n_if, result_if = get_TOS_iforest(X, y, n_range,
                                                            feature_list)

# Generate TOS using isolation forests
feature_list, roc_hbos, prc_n_hbos, result_hbos = get_TOS_hbos(X, y, k_range,
                                                            feature_list)
##############################################################################
# combine the feature space by concanating various TOS
X_train_new_orig = np.concatenate(
    (result_knn, result_loop, result_lof, result_ocsvm, result_if), axis=1)

X_train_all_orig = np.concatenate((X, X_train_new_orig), axis=1)

# combine ROC and Precision@n list
roc_list = roc_knn + roc_loop + roc_lof + roc_ocsvm + roc_if
prc_n_list = prc_n_knn + prc_n_loop + prc_n_lof + prc_n_ocsvm + prc_n_if

# get the results of baselines
print_baseline(X_train_new_orig, y, roc_list, prc_n_list)

##############################################################################
# select TOS using different methods

p = 10  # number of selected TOS

# random selection
# please be noted the actual random selection happens within the
# train-test split, with p repetitions.
X_train_new_rand, X_train_all_rand = random_select(X, X_train_new_orig,
                                                   roc_list, p)
# accurate selection
X_train_new_accu, X_train_all_accu = accurate_select(X, X_train_new_orig,
                                                     roc_list, p)
# balance selection
X_train_new_bal, X_train_all_bal = balance_select(X, X_train_new_orig,
                                                  roc_list, p)

###############################################################################
# build various classifiers

# it is noted that the data split should happen as the first stage
# test data should not be exposed. However, with a relatively large number of
# repetitions, the demo code would generate a similar result.

# the full code uses the containers to save the intermediate TOS models. The
# codes would be shared after the cleanup.

ite = 30  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%
result_dict = {}

clf_list = [XGBClassifier(), LogisticRegression(penalty="l1"),
            LogisticRegression(penalty="l2")]
clf_name_list = ['xgb', 'lr1', 'lr2']

# initialize the result dictionary
for clf_name in clf_name_list:
    result_dict[clf_name + 'ROC' + 'o'] = []
    result_dict[clf_name + 'ROC' + 's'] = []
    result_dict[clf_name + 'ROC' + 'n'] = []

    result_dict[clf_name + 'PRC@n' + 'o'] = []
    result_dict[clf_name + 'PRC@n' + 's'] = []
    result_dict[clf_name + 'PRC@n' + 'n'] = []

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
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'ROC' + 'o'].append(roc_score)
        result_dict[clf_name + 'PRC@n' + 'o'].append(prec_n)

        # unsupervised
        clf.fit(X_train_n, y_train.ravel())
        y_pred = clf.predict_proba(X_test_n)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'ROC' + 'n'].append(roc_score)
        result_dict[clf_name + 'PRC@n' + 'n'].append(prec_n)

        # semi-supervised
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict_proba(X_test)

        roc_score = roc_auc_score(y_test, y_pred[:, 1])
        prec_n = get_precn(y_test, y_pred[:, 1])

        result_dict[clf_name + 'ROC' + 's'].append(roc_score)
        result_dict[clf_name + 'PRC@n' + 's'].append(prec_n)

for eva in ['ROC', 'PRC@n']:
    print()
    for clf_name in clf_name_list:
        print(np.round(np.mean(result_dict[clf_name + eva + 'o']), decimals=4),
              eva, clf_name, 'original features')
        print(np.round(np.mean(result_dict[clf_name + eva + 'n']), decimals=4),
              eva, clf_name, 'TOS only')
        print(np.round(np.mean(result_dict[clf_name + eva + 's']), decimals=4),
              eva, clf_name, 'original feature + TOS')
