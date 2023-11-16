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
        activations = [X]

        for i in range(len(self.weights) - 1):
            print(f'Layer: {i}, x: {activations[i].shape}, w: {self.weights[i].shape}, b: {self.bias[i].shape}')
            linear_output = np.dot(activations[i], self.weights[i]) + self.bias[i]
            print(f'Linear part output shape: {linear_output.shape}')
            activation_output = self.relu(linear_output)
            print(f'Activation part output shape: {activation_output.shape}')
            activations.append(activation_output)
        
        # output layer
        last_linear_output = np.dot(activations[-1], self.weights[-1]) + self.bias[-1]
        print(f'Last linear part output shape: {last_linear_output.shape}')
        last_activation_output = self.sigmoid(last_linear_output)
        print(f'Last activation part output shape: {last_activation_output.shape}')
        activations.append(last_activation_output)

        return activations
    
    # back propagation
    def back_propagation(self, X, y, activations):
        # calculate the error of the output layer
        error = activations[-1] - y
        # calculate the error of the hidden layers
        for i in range(len(self.weights)-1, 0, -1):
            error = np.dot(error, self.weights[i].T) * self.relu_prime(activations[i])
            self.weights[i] -= self.eta * np.dot(activations[i-1].T, error)
            self.bias[i] -= self.eta * np.sum(error, axis=0, keepdims=True)
        
        # update the weights and bias of the input layer
        error = np.dot(error, self.weights[1].T) * self.relu_prime(activations[0])
        self.weights[0] -= self.eta * np.dot(X.T, error)
        self.bias[0] -= self.eta * np.sum(error, axis=0, keepdims=True)


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



    



    