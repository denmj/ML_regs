"""
    Multilayer Perceptron

"""

import numpy as np


class MultilayerPerceptron(object):
    """
    Multilayer Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Passes over the training dataset.
    hidden_layers : list
        Number of hidden layers and number of neurons in each layer.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    errors_ : list
        Number of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, epochs=50, hidden_layers=[10], shuffle=True, random_state=None):
        self.eta = eta
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.shuffle = shuffle
        self.random_state = random_state


    