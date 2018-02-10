import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def get_precn(y, y_pred):
    '''
    Utlity function to calculate precision@n
    :param y: ground truth
    :param y_pred: number of outliers
    :return: score
    '''
    # calculate the percentage of outliers
    out_perc = np.count_nonzero(y) / len(y)

    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return precision_score(y, y_pred)


def precision_n(y_pred, y, n):
    '''
    Utlity function to calculate precision@n

    :param y_pred: predicted value
    :param y: ground truth
    :param n: number of outliers
    :return: scaler score
    '''
    y_pred = np.asarray(y_pred)
    y = np.asarray(y)

    length = y.shape[0]

    assert (y_pred.shape == y.shape)
    y_sorted = np.partition(y_pred, int(length - n))

    threshold = y_sorted[int(length - n)]

    y_n = np.greater_equal(y_pred, threshold).astype(int)
    #    print(threshold, y_n, precision_score(y, y_n))

    return precision_score(y, y_n)


def get_top_n(roc_list, n, top=True):
    '''
    for use of Accurate Selection only
    :param roc_list: a li
    :param n:
    :param top:
    :return:
    '''
    roc_list = np.asarray(roc_list)
    length = roc_list.shape[0]

    roc_sorted = np.partition(roc_list, length - n)
    threshold = roc_sorted[int(length - n)]

    if top:
        return np.where(np.greater_equal(roc_list, threshold))
    else:
        return np.where(np.less(roc_list, threshold))


def print_baseline(X_train_new_orig, y, roc_list, prec_list):
    max_value_idx = roc_list.index(max(roc_list))
    print()
    print('Highest TOS ROC:', roc_list[max_value_idx])
    print('Highest TOS Precison@n', max(prec_list))

    # normalized score
    X_train_all_norm = StandardScaler().fit_transform(X_train_new_orig)
    X_train_all_norm_mean = np.mean(X_train_all_norm, axis=1)

    roc = np.round(roc_auc_score(y, X_train_all_norm_mean), decimals=4)
    prec_n = np.round(get_precn(y, X_train_all_norm_mean), decimals=4)

    print('Average TOS ROC:', roc)
    print('Average TOS Precision@n', prec_n)
