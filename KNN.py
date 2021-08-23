import numpy as np
"""
K-nearest Neighbors
"""

class KNN():
    def __init__(self, k):
        self.k_ = k

    def _computeDist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def fit(self, x, y, test):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x_s = (x - mean) / std
        test_s = (test - x_s) / std

        m, n = x_s.shape[0], test_s.shape[0]
        res = []
        for i in range(n):
            cur = test_s[i, :]
            tmp = []
            for j in range(m):
                compare = x_s[j, :]
                dist = self._computeDist(cur, compare)
                tmp.append(dist)
            tmp = np.array(tmp)
            top_k_dix = tmp.argsort()[:self.k_]
            pred = y[top_k_dix]
            pred_mean = (1 / self.k_) * np.sum(pred)
            res.append(pred_mean)
        return res

