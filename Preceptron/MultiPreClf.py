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

    def __init__(self, alpha=0.01, 
                 n_iterations=50,
                 input_layers=784, 
                 hidden_layers=[128, 64],
                 n_outputs= 10):
        
        # alpha is the learning rate
        self.eta = alpha

        # n_iterations is the number of training iterations
        self.n_iterations = n_iterations

        # input_layers is usually the number of features (MNIST: 784)
        self.input_layers = input_layers

        # n_outputs is the number of output neurons (MNIST: 10)
        self.n_outputs = n_outputs

        self.weights = []
        self.bias = []
        
        # loop initialization
        layer_sizes = [input_layers] + hidden_layers + [n_outputs]

        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.bias.append(np.zeros((1, layer_sizes[i+1])))



    



    