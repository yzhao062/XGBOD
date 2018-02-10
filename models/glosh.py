import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.utility import get_precn


class Glosh(object):
    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size

    def fit(self, X_train):
        self.X_train = X_train

    def sample_scores(self, X_test):
        # initialize the outputs
        pred_score = np.zeros([X_test.shape[0], 1])

        for i in range(X_test.shape[0]):
            x_i = X_test[i, :]

            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])
            x_comb = np.concatenate((self.X_train, x_i), axis=0)

            x_comb_norm = StandardScaler().fit_transform(x_comb)

            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(x_comb_norm)

            # print(clusterer.outlier_scores_[-1])
            # record the current item
            pred_score[i, :] = clusterer.outlier_scores_[-1]
        return pred_score

    def evaluate(self, X_test, y_test):
        pred_score = self.sample_scores(X_test)
        prec_n = (get_precn(y_test, pred_score))

        print("precision@n", prec_n)
