import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import scoreatpercentile

class Hbos(object):

    def __init__(self, bins=10, alpha=0.3, beta=0.5, contamination=0.05):

        self.bins = bins
        self.alpha = alpha
        self.beta = beta
        self.contamination = contamination

    def fit(self, X):

        self.n, self.d = X.shape[0], X.shape[1]
        out_scores = np.zeros([self.n, self.d])

        hist = np.zeros([self.bins, self.d])
        bin_edges = np.zeros([self.bins + 1, self.d])

        # this is actually the fitting
        for i in range(self.d):
            hist[:, i], bin_edges[:, i] = np.histogram(X[:, i], bins=self.bins,
                                                       density=True)
            # check the integrity
            assert (
                math.isclose(np.sum(hist[:, i] * np.diff(bin_edges[:, i])), 1))

        # calculate the threshold
        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X[:, i], bin_edges[:, i], right=False)

            # very important to do scaling. Not necessary to use min max
            density_norm = MinMaxScaler().fit_transform(
                hist[:, i].reshape(-1, 1))
            out_score = np.log(1 / (density_norm + self.alpha))

            for j in range(self.n):
                # out sample left
                if bin_ind[j] == 0:
                    dist = np.abs(X[j, i] - bin_edges[0, i])
                    bin_width = bin_edges[1, i] - bin_edges[0, i]
                    # assign it to bin 0
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j]]
                    else:
                        out_scores[j, i] = np.max(out_score)

                # out sample right
                elif bin_ind[j] == bin_edges.shape[0]:
                    dist = np.abs(X[j, i] - bin_edges[-1, i])
                    bin_width = bin_edges[-1, i] - bin_edges[-2, i]
                    # assign it to bin k
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j] - 2]
                    else:
                        out_scores[j, i] = np.max(out_score)
                else:
                    out_scores[j, i] = out_score[bin_ind[j] - 1]

        out_scores_sum = np.sum(out_scores, axis=1)
        self.threshold = scoreatpercentile(out_scores_sum,
                                           100 * (1 - self.contamination))
        self.hist = hist
        self.bin_edges = bin_edges
        self.decision_scores = out_scores_sum
        self.y_pred = (self.decision_scores > self.threshold).astype('int')

    def decision_function(self, X_test):

        n_test = X_test.shape[0]
        out_scores = np.zeros([n_test, self.d])

        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X_test[:, i], self.bin_edges[:, i],
                                  right=False)

            # very important to do scaling. Not necessary to use minmax
            density_norm = MinMaxScaler().fit_transform(
                self.hist[:, i].reshape(-1, 1))

            out_score = np.log(1 / (density_norm + self.alpha))

            for j in range(n_test):
                # out sample left
                if bin_ind[j] == 0:
                    dist = np.abs(X_test[j, i] - self.bin_edges[0, i])
                    bin_width = self.bin_edges[1, i] - self.bin_edges[0, i]
                    # assign it to bin 0
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j]]
                    else:
                        out_scores[j, i] = np.max(out_score)

                # out sample right
                elif bin_ind[j] == self.bin_edges.shape[0]:
                    dist = np.abs(X_test[j, i] - self.bin_edges[-1, i])
                    bin_width = self.bin_edges[-1, i] - self.bin_edges[-2, i]
                    # assign it to bin k
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j] - 2]
                    else:
                        out_scores[j, i] = np.max(out_score)
                else:
                    out_scores[j, i] = out_score[bin_ind[j] - 1]

        out_scores_sum = np.sum(out_scores, axis=1)
        return out_scores_sum

    def predict(self, X_test):
        pred_score = self.decision_function(X_test)
        return (pred_score > self.threshold).astype('int')
