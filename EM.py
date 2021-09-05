import numpy as np

class EM():
    """
    Before EM, give initial values for learnable parameters
    Initialize the weight parameters as 1/k (k is the # of categories) and guess the values for the means and variances
    Then start EM algorithm. E stands for expectation and M stands for maximization

    E Step: in the E step, we calculate the likelihood of each observation x_i using the estimated parameters (pdf)
    Then we can calculate the likelihood of a given example x_i to belong to the k_th cluster using
                                            b_k = f * weight / sum(f * weight)
    M Step: in the M step, we re-estimate our learning parameters:
    mean = sum(b_k * x) / sum(b_k)   variance=sum(b_k * (x-mean)^2)/sum(b_k)    weight=sum(b_k)/N
    repeat these steps until convergence
    """

    def __init__(self, k, input, iterations):
        self.X = input
        self.k = k
        self.iterations = iterations
        self.weights = np.ones((k)) / k
        self.means = np.random.choice(self.X, k)
        self.variances = np.random.random_sample(size=k)

    def fit(self):
        for _ in range(self.iterations):
            self._expectation()
            self._maximization()
        return self.weights, self.means, self.variances

    def _expectation(self):
        likelihood = []
        for i in range(self.k):
            likelihood.append(self.pdf(self.X, self.means[i], self.variances[i]))
        self.likelihood = np.array(likelihood)


    def _maximization(self):
        b = np.zeros((self.k, self.X.shape[0]))
        for i in range(self.X.shape[0]):
            b_ = self.weights * self.likelihood[:, i]
            sum_b = np.sum(b_)
            b[:, i] = b_ / sum_b

        for i in range(self.k):
            self.means[i] = np.sum(b[i, :] * self.X) / np.sum(b[i, :])
            self.variances[i] = np.sum(b[i, :] * (self.X - self.means[i]) ** 2) / np.sum(b[i, :])
            self.weights[i] = np.sum(b[i, :]) / self.X.shape[0]


    def pdf(self, data, mean, variance):
        constant = 1 / (np.sqrt(2 * np.pi * variance))
        gauss = constant * np.exp(-(data - mean) ** 2 / (2 * variance))
        return gauss