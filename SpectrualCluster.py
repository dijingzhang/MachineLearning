import numpy as np
from sklearn.cluster import KMeans

class SpectrualCluster():
    def _getSimilarity(self, arrA, arrB, sigma):
        a = -(np.sum(np.square(arrA - arrB)))
        b = 2 * np.square(sigma)
        s = np.exp(a / b)
        return s

    def _getAdjacencyMatrix(self, points, sigma):
        n = points.shape[0]
        W = np.zeros((n, n))
        for i in range(n):
            x = points[i, :]
            for j in range(n):
                y = points[j, :]
                W[i, j] = self._getSimilarity(x, y, sigma)
        return W - np.identity(n)

    def _getLaplacianMatrix(self, points, sigma):
        W = self._getAdjacencyMatrix(points, sigma)
        D_sqrt = np.mat(np.diag(np.power(np.sum(W, axis=1), -0.5)))
        L = D_sqrt * np.mat(W) * D_sqrt
        return L

    def _standardize(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / std

    def fit(self, points, sigma, k, n):
        L = self._getLaplacianMatrix(points, sigma)
        w, v = np.linalg.eig(L)
        index = np.argsort(w)[-k:]
        X = v[:, index]
        X = self._standardize(X)

        # kmeans
        kmeans = KMeans(n_clusters=n).fit(X)
        prediction = kmeans.labels
        return prediction







