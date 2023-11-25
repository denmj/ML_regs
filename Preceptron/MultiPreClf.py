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
                 n_outputs= 10,
                 regularization='l2',
                 lambda_reg=0.01):
        
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
            # He initialization
            he_std_dev = np.sqrt(2 / self.layer_sizes[i])

            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * he_std_dev)
            self.bias.append(np.zeros((1, self.layer_sizes[i+1])))
        # print shape of weights
        for i in range(len(self.weights)):
            print(f'Weights shape: {self.weights[i].shape}')
    
    # feed forward
    def forward_propagation(self, X):
        activations = [X]
        # -1 because the last layer is the output layer
        for i in range(len(self.weights) - 1):
            linear_output = np.dot(activations[i], self.weights[i]) + self.bias[i]
            activation_output = self.relu(linear_output)
            # print activation[i].shape, weights[i].shape, linear_output.shape

            activations.append(activation_output)
        # output layer
        last_linear_output = np.dot(activations[-1], self.weights[-1]) + self.bias[-1]

        # check if case is binary or multi-class classification
        if self.n_outputs == 1:
            last_activation_output = self.sigmoid(last_linear_output)
        else:
            last_activation_output = self.softmax(last_linear_output)
    
        activations.append(last_activation_output)

        return activations
    
    # back propagation
    def back_propagation(self, X, y, activations):
        
        for layer in range (1, len(self.weights) + 1):

            if -layer == -1:

                if self.n_outputs == 1:
                    error = activations[-1] - y * self.sigmoid_prime(activations[-1])
                else:
                    error = activations[-1] - y
            else:
                error = np.dot(error, self.weights[-layer+1].T) * self.relu_prime(activations[-layer])

            if self.regularization == 'l2':
                reg_penalty = self.lambda_reg * self.weights[-layer]
            elif self.regularization == 'l1':
                reg_penalty = self.lambda_reg * np.sign(self.weights[-layer])
            else:
                reg_penalty = 0

            delta = error * self.eta

            self.weights[-layer] -=  (np.dot(activations[-layer-1].T, delta) + reg_penalty)
            self.bias[-layer] -= np.sum(delta, axis=0, keepdims=True)

    def train(self, X, y, X_val = None, y_val = None, batch_size = 32):
        
        training_loss = []
        validation_loss = []

        for i in range(self.n_iterations):
            print(f'Iteration: {i}')
            # mini-batch
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for j in range(0, X.shape[0], batch_size):

                X_batch = X_shuffled[j:j+batch_size]
                y_batch = y_shuffled[j:j+batch_size]

                activations = self.forward_propagation(X_batch)
                self.back_propagation(X_batch, y_batch, activations)

            # calculate the loss
            loss = self.cross_entropy_loss(y, self.forward_propagation(X)[-1])

            print(f'Epoch: {i}, Loss: {loss}')
            training_loss.append(loss)

            if X_val is not None and y_val is not None:
      
                val_loss = self.cross_entropy_loss(y_val, self.forward_propagation(X_val)[-1])
                print(f'Epoch: {i}, Loss: {val_loss}')
                validation_loss.append(val_loss)
        
        return training_loss, validation_loss


    # Loss calculation - MSE
    def calculate_mse_loss(self, y, y_pred):
        return np.mean((y - y_pred)**2)
    
    # Loss calculation - Cross Entropy
    def cross_entropy_loss(self, y_true, y_pred):
        # Assuming y_pred is output from sigmoid (binary) or softmax (multi-class)
        if y_true.shape[1] == 1:  # Binary classification
            return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        else:  # Multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))



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
    
    # activation function - softmax
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    # save model weights
    def save_weights(self, filename):
        np.savez(filename, *self.weights)

    # load model weights
    def load_weights(self, filename):
        weights = np.load(filename)
        for i in range(len(self.weights)):
            self.weights[i] = weights['arr_%d' % i]

    # Accuracy calculation
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred, axis=0) / len(y_true)

    def recall(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / np.sum(y_true)

    def precision(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / np.sum(y_pred)
    
    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * p * r / (p + r)
    



    