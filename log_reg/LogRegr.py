import numpy as np
from sklearn.preprocessing import OneHotEncoder
from helper_funcs import *


class LogisticRegression_manual:
    def __init__(self, x_train, y_train, iterations, learning_rate, multiclass = False, classes=1):

        self.x_train = x_train
        self.y_train = y_train

        self.EPOCHS = iterations
        self.ALPHA = learning_rate
        self.classes = classes

        self.multiclass = multiclass

        self.grad_and_cost()
        self.init_weights()


        pass

    def init_weights(self):
        self.w = np.zeros([self.x_train.shape[1], self.classes])
        self.b = np.zeros([1, self.classes])
        return self.w, self.b

    def grad_and_cost(self):

        """
        w - weights, a
        np
        array
        of
        size(features, 1)
        b - bias, a
        scalar

        """
        w, b = self.init_weights()
        if not len(self.x_train.shape) == 2:
            X_flattened = self.x_train.reshape(self.x_train.shape[1]*self.x_train.shape[2], -1).T
        else:
            X_flattened = self.x_train
        m = X_flattened.shape[1]

        if self.multiclass:
            # Multi-class

            y_train_reshaped = self.y_train.reshape(len(self.y_train), 1)
            ohe = OneHotEncoder()
            y_train_reshaped = ohe.fit_transform(y_train_reshaped).toarray()
            A = softmax(np.dot(X_flattened, w) + b)
            xentropy = -np.sum(y_train_reshaped * np.log(A))
            cost = np.mean(-1 / m * np.sum(y_train_reshaped*np.log(A)+(1-y_train_reshaped)*np.log(1-A), axis=1, keepdims=True))
            dw = 1 / m * np.dot(X_flattened.T, (A - y_train_reshaped))
            db = 1 / m * np.sum(A - y_train_reshaped)
        else:
            # Binary
            A = sigmoid(np.dot(w.T, X_flattened) + b)
            cost = -1 / m * np.sum(self.y_train*np.log(A)+(1-self.y_train)*np.log(1-A), axis=1, keepdims=True)
            dw = 1 / m * np.dot(X_flattened, (A - self.y_train).T)
            db = 1 / m * np.sum(A - self.y_train)

        # grads/derivatives
        cost = np.squeeze(cost)

        return dw, db, cost

    def fit(self):
        costs = []

        for epoch in range(self.EPOCHS):

            dw, db, cost = self.grad_and_cost()

            w = w - self.ALPHA * dw
            b = b - self.ALPHA * db
            if epoch % 100 == 0:
                costs.append(cost)
                print(cost)

        return costs, w, b

    def predict(self):
        pass