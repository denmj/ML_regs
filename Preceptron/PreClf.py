import numpy as np


class PreceptClf:
    def __init__(self, alpha=0.01, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.cost_history = []

        self.weights = None
        self.bias = None
        self.activation_func = self.sign_func

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_cond = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iterations):
            cost_sum = 0.0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Update
                update = self.alpha * (y_cond[idx] - y_predicted)
                cost = (y_cond[idx] - y_predicted) ** 2

                cost_sum += cost
                self.weights += update * x_i
                self.bias += update
            self.cost_history.append(cost_sum)


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def sign_func(self, X):
        return np.where(X >= 0, 1, 0)