import numpy as np


class SingleLayerPreceptronClassifier:

    """
        Silgle Layer Preceptron Classifier
        Parameters:
        ------------
        alpha: float
            Learning rate (between 0.0 and 1.0)

        n_iterations: int
            Number of passes over the training dataset.
        
        Attributes:
        ------------
        weights: 1d-array
            Weights after fitting.

        bias: float
            Bias after fitting.
        
        cost_history: list
            List of costs after each iteration.

    """

    def __init__(self, alpha=0.01, n_iterations=1000):
        # alpha is the learning rate
        self.alpha = alpha
        # n_iterations is the number of training iterations
        self.n_iterations = n_iterations
        # weights and bias are the model parameters
        self.cost_history = []

        # weights and bias are the model parameters
        self.weights = None
        self.bias = None

        # activation function is the sigmoid function
        self.activation_func = self.sign_func

    def fit(self, X, y):

        """
            Fit training data.
            Parameters:
            ------------
            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples 
                and n_features is the number of features.
            
            y: array-like, shape = [n_samples]
                Target values.
            
            Returns:
            ------------
            None
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        
        for _ in range(self.n_iterations):
            cost_sum = 0.0
            for x_i, target in zip(X, y):

                # Linear output of single neuron
                linear_output = np.dot(x_i, self.weights) + self.bias

                # Predicted output of single neuron
                y_predicted = self.activation_func(linear_output)

                # Update
                update = self.alpha * (target - y_predicted)
                self.weights += update * x_i
                self.bias += update

                cost = (target - y_predicted) ** 2
                cost_sum += cost
            self.cost_history.append(cost_sum)


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def sign_func(self, X):
        return np.where(X >= 0, 1, 0)