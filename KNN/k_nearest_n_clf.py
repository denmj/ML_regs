import numpy as np


class KNearestNeighbors:

    def __init__(self, X, y):

        # Remember all data (training)
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """

        :param X: A numpy array X_test with shape (num_test, m)
        :param k: number of neighbors
        :return: returs distances between test and train examples
        """

        distances = self.compute_disctances(X)

        return self.predict_labels(distances, k)

    def compute_disctances(self, X):
        """

        :param X: A numpy array X_test with shape (num_test, m)
        :return:
        """
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                distances[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        return distances

    def predict_labels(self, distances, k=1):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            closest_y = [self.y_train[np.argsort(distances[i])[:k]]]

            unique, counts = np.unique(closest_y[-1], return_counts=True)
            vals_cnt = np.array([unique, counts]).T
            vals_cnt_sorted = vals_cnt[vals_cnt[:, 1].argsort()[::-1]]
            y_pred[i] = vals_cnt_sorted[0][0]

        return y_pred


X = np.random.rand(10, 3)
W = np.zeros((X.shape[1], 2))
print(X.shape, W.shape)

z = np.dot(X, W)
print(z)