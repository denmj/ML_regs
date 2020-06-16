import numpy as np
from sklearn import datasets

np.random.seed(44)
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)
weights = 0.001 * np.random.randn(X.shape[1])


def comp_cost(W, X, Y, reg):
    # hinge loss 1/2 * W^2 + 1/N sum(max(0, 1-y(w*x +b)
    N = X.shape[0]

    distances = 1 - Y * (np.dot(X, W))

    distances[distances < 0] = 0

    hinge_loss = reg * (np.sum(distances) / N)
    loss = 1/2 * np.dot(W, W) + hinge_loss
    return loss


def comp_gradient(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - y * (np.dot(X_batch, W))
    dW = np.zeros(len(W))
    for index, d in enumerate(distance):
        if max(0, d) == 0:
            di = weights
        else:
            di = weights - (0.001 * y[index] * X[index])
        dW += di
    dw = dW / len(y)
    return dw


def sgd(X, Y):
    pass


class SupportVectorMachine:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iter=100,
              batch_size=200, verbose=False):
        """

        :param X:
        :param y:
        :param learning_rate:
        :param reg:
        :param num_iter:
        :param batch_size:
        :param verbose:
        :return:
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_hist = []
        for it in range(num_iter):

            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(range(X.shape[0]), size=10)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grad = self.loss(X_batch, y_batch, reg)

            self.W += -learning_rate*grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iter, loss))

    def predict(self, X):
        pass

    def loss(self, X_batch, y_batch, reg):
        loss = 0
        dW = np.zeros(self.W.shape)
        num_train = X.shape[0]
        return 0



