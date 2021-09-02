import numpy as np
from cvxopt import matrix, solvers

class SVM():
    def __init__(self, points, label, C=None):
        """
        :param points: [n, 2] 2D coordinates and have n samples
        :param label: [n, ] n labels
        :param C: hyperparameters to control overfit, larger will cause SVM closer to hard margin
        """
        self.points = points
        self.label = label
        self.C = C

    def fit(self):
        if self.C is None:
            return self.linearSVM()
        else:
            return self.nonlinearSVM()

    def linearSVM(self):
        x, y = self.points, self.label
        n = x.shape[0]
        Q = np.zeros((3, 3))
        Q[0, 0], Q[1, 1] = 1, 1
        h = -np.zeros((n, 1))
        p = np.zeros((3, 1))
        x = np.append(x, np.ones((n, 1)), axis=1)
        G = - x * y.reshape((n, 1))
        sol = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
        z = sol['x']
        margin = 1 / (np.sqrt(z[0] ** 2 + z[1] ** 2))
        print("margin: ", margin)
        return z

    def nonlinearSVM(self):
        x, y = self.points, self.label
        n = x.shape[0]
        Q = np.zeros((n+3, n+3))
        Q[0][0], Q[1][1] = 1, 1
        p = np.zeros(n+3)
        p[3:] = self.C
        h = - np.ones((2 * n))
        h[n:] = 0
        G = np.zeros((2*n, n+3))
        x = np.append(x, np.ones((n, 1)), axis=1)
        G[:n, :3] = - x * y.reshape((n, 1))
        G[:n, 3:] = -np.diag(np.ones(n))
        G[n:, 3:] = -np.diag(np.ones(n))

        sol = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
        z = sol['x']
        margin = 1 / (np.sqrt(z[0] ** 2 + z[1] ** 2))
        print("margin: ", margin)

        return z