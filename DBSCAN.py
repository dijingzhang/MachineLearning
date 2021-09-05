import numpy as np



class DBSCAN():
    UNCLASSIFIED = False
    NOISE = 0

    def __init__(self, minPts, eps):
        self.eps = eps
        self.minPts = minPts

    def _dist(self, a, b):
        return np.sqrt(np.power(a - b, 2).sum())

    def _eps_neighbor(self, a, b):
        return self._dist(a, b) < self.eps

    def _region_query(self, data, pointId):
        """
        :param data: dataset
        :param pointId: id of query point
        :return: the id within eps
        """
        nPoints = data.shape[0]
        seeds = []
        for i in range(nPoints):
            if self._eps_neighbor(data[pointId, :], data[i, :]):
                seeds.append(i)
        return seeds

    def _expand_cluster(self, data, clusterResult, pointId, clusterId):
        """
        :param clusterResult: the result of cluster
        :param clusterId: the id of cluster
        :return: whether the point can be classified
        """

        seeds = self._region_query(data, pointId)
        if len(seeds) < self.minPts:
            clusterResult[pointId] = self.NOISE
            return False
        clusterResult[pointId] = clusterId
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0:
            currentPoint = seeds[0]
            queryResults = self._region_query(data, currentPoint)
            if len(queryResults) >= self.minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == self.UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == self.NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True

    def fit(self, data):
        clusterId = 1
        nPoints = data.shape[0]
        clusterResult = [self.UNCLASSIFIED] * nPoints
        for pointId in range(nPoints):
            if clusterResult[pointId] == self.UNCLASSIFIED:
                if self._expand_cluster(data, clusterResult, pointId, clusterId):
                    clusterId += 1
        return clusterResult, clusterId - 1




