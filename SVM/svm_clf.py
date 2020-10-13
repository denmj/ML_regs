import numpy as np
from sklearn import datasets

# rand data set
np.random.seed(44)
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)
weights = 0.001 * np.random.randn(X.shape[1])


# Binary case
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

            loss = self.comp_cost(self.W, X_batch, y_batch, 0.001)
            loss_hist.append(loss)
            grad = self.comp_grad(self.W, X_batch, y_batch, 0.001)

            self.W += -learning_rate*grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iter, loss))
        return loss_hist

    def comp_cost(self, W,  X, Y, reg):

        self.W = W
        training_examples = X[0].shape

        # hinge loss 1/2 * W^2 + 1/N sum(max(0, 1-y(w*x +b)
        distances = 1 - Y * (np.dot(X, self.W))

        distances[distances < 0] = 0

        hinge_loss = reg * (np.sum(distances) / training_examples)
        loss = 1/2 * np.dot(self.W, self.W) + hinge_loss
        return loss

    def comp_grad(self, W, X_batch, Y_batch, reg):

        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])

        distance = 1 - y * (np.dot(X_batch, self.W))
        dW = np.zeros(len(W))
        for index, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (0.001 * y[index] * X[index])
            dW += di
        dw = dW / len(y)
        return dw

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(np.dot(X, self.W), axis=1)
        return y_pred

