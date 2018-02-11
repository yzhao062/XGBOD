import random
import numpy as np
from scipy.stats import pearsonr

from models.utility import get_top_n


def random_select(X, X_train_new_orig, roc_list, p):
    s_feature_rand = random.sample(range(0, len(roc_list)), p)
    X_train_new_rand = X_train_new_orig[:, s_feature_rand]
    X_train_all_rand = np.concatenate((X, X_train_new_rand), axis=1)

    # print(s_feature_rand)

    return X_train_new_rand, X_train_all_rand


def accurate_select(X, X_train_new_orig, roc_list, p):
    s_feature_accu = get_top_n(roc_list=roc_list, n=p, top=True)
    X_train_new_accu = X_train_new_orig[:, s_feature_accu[0][0:p]]
    X_train_all_accu = np.concatenate((X, X_train_new_accu), axis=1)

    # print(s_feature_accu)

    return X_train_new_accu, X_train_all_accu


def balance_select(X, X_train_new_orig, roc_list, p):
    s_feature_balance = []
    pearson_list = np.zeros([len(roc_list), 1])

    # handle the first value
    max_value_idx = roc_list.index(max(roc_list))
    s_feature_balance.append(max_value_idx)
    roc_list[max_value_idx] = -1

    for i in range(p - 1):

        for j in range(len(roc_list)):
            pear = pearsonr(X_train_new_orig[:, max_value_idx],
                            X_train_new_orig[:, j])

            # update the pearson
            pearson_list[j] = np.abs(pearson_list[j]) + np.abs(pear[0])

        discounted_roc = np.true_divide(roc_list, pearson_list.transpose())

        max_value_idx = np.argmax(discounted_roc)
        s_feature_balance.append(max_value_idx)
        roc_list[max_value_idx] = -1

    X_train_new_balance = X_train_new_orig[:, s_feature_balance]
    X_train_all_balance = np.concatenate((X, X_train_new_balance), axis=1)

    # print(s_feature_balance)

    return X_train_new_balance, X_train_all_balance
