import numpy as np


class LinReg:
    def __init__(self, alpha=0.01, n_iterations=1200):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.grad_history = []
        self.weights_history = []
        self.bias_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros([n_features, 1])
        self.weights_history.append(self.weights)
        self.bias = np.ones([n_samples, 1])
        self.bias_history.append(self.bias[0])
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # gradients
            dW = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update
            self.weights -= self.alpha * dW
            self.bias -= self.alpha * db
            cost = 1 / (2 * n_samples) * sum((y_predicted - y) ** 2)
            self.cost_history.append(cost)
            self.weights_history.append(self.weights)
            self.bias_history.append(self.bias[0])
            self.grad_history.append((self.weights, self.bias))

    def predict(self, X):
        y_approx = np.dot(X, self.weights) + self.bias
        return y_approx
