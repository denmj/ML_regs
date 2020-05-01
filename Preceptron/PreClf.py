import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.make_regression(n_samples=10, n_features=2, noise=10, random_state=40)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
y = y.reshape(10, 1)
w = np.zeros([x_train.shape[1], 1])

def sign_func(z):
    if z > 0:
        return 1
    else:
        return -1

for i, x in enumerate(x_train):
    print(sign_func(np.dot(x, w)))


class PreceptClf:
    def __init__(self, alpha=0.01, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros([n_features, 1])
        self.bias = np.ones([n_samples, 1])

    def predict(self):
        pass