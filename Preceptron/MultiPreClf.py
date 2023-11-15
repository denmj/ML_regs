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
                 input_layer_size=784, 
                 hidden_layers_size=[128, 64],
                 n_outputs= 10):
        
        # alpha is the learning rate
        self.eta = alpha

        # n_iterations is the number of training iterations
        self.n_iterations = n_iterations

        # input_layers is usually the number of features (MNIST: 784)
        self.input_layer_size = input_layer_size

        # hidden_layers is a list of the number of neurons in each hidden layer
        self.hidden_layers_size = hidden_layers_size

        # n_outputs is the number of output neurons (MNIST: 10)
        self.n_outputs = n_outputs

        self.weights = []
        self.bias = []
        
        # loop initialization
        self.layer_sizes = [self.input_layer_size] + self.hidden_layers_size + [self.n_outputs]

        for i in range(len(self.layer_sizes)-1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]))
            self.bias.append(np.zeros((1, self.layer_sizes[i+1])))

    
    # feed forward
    def forward_propagation(self, X):
        pass


    # activation function - sigmoid 
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    # derivative of sigmoid function 
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    # activation function - relu
    def relu(self, z):
        return np.maximum(0, z)

    # derivative of relu function
    def relu_prime(self, z):
        return (z > 0).astype(int)



    



    