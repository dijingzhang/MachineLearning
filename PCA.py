import numpy as np


class DimensionValueError(ValueError):
    pass


class PCA(objects):
    def __init__(self, x, n_components=None):
        """
        :param x: [n, m], there are m n-dimension data
        :param n_components: the reduced dimensionality
        """

        self.x = x
        self.dimension = x.shape[0]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def _mean(self):
        mean = np.mean(self.x, axis=1)
        self.x -= mean
        return mean

    def _cov(self):
        X = self.x
        cov = (X @ X.T) / m
        return cov

    def _get_feature(self):
        """
        :return: P: [k, n]
        """
        cov = self._cov()
        w, v = np.linalg.eig(cov)
        sorting_index = np.argsort(-w)
        picked_index = sorting_index[:self.n_components]
        P = v[:, picked_index].T
        return P


    def reduce_dimension(self):
        mean = self._mean()
        P = self._get_feature()
        reduced_dimension_data = P @ self.x # [k, m]
        return reduced_dimension_data, mean




