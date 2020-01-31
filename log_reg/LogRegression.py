import numpy as np
from sklearn.preprocessing import OneHotEncoder


class LogisticRegression:
    def __init__(self, x_train, y_train, iterations, learning_rate, muliclass = False):

        self.x_train = x_train
        self.y_train = y_train

        self.EPOCHS = iterations
        self.ALPHA = learning_rate

        self.multiclass = muliclass


        pass

    def fit(self):
        pass

    def predict(self):
        pass