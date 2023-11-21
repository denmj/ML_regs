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

        for i in range(len(self.weights) - 1):
            linear_output = np.dot(activations[i], self.weights[i]) + self.bias[i]
            activation_output = self.relu(linear_output)
            activations.append(activation_output)
        
        # output layer
        last_linear_output = np.dot(activations[-1], self.weights[-1]) + self.bias[-1]

        # check if case is binary or multi-class classification
        if self.n_outputs == 1:
            last_activation_output = self.sigmoid(last_linear_output)
        else:
            last_activation_output = self.softmax(last_linear_output)

        activations.append(last_activation_output)
        # print shape of activations
        for i in range(len(activations)):
            print(f'Activations shape: {activations[i].shape}')

        return activations
    
    # back propagation
    def back_propagation(self, X, y, activations):
        
        # initial error 
        output_error = activations[-1] - y

        #Backward pass 
        for l in range(1, len(self.weights) + 1):
            # index from the end of the list
            i = -l

            # calculate gradients
            if i == -1:
                # check if case is binary or multi-class classification
                if self.n_outputs == 1:
                    error = output_error * self.sigmoid_prime(activations[i])
                else:
                    error = output_error * self.softmax(activations[i])
            else:
                error = np.dot(delta, self.weights[i+1].T) * self.relu_prime(activations[i])

            # calculate delta
            delta = error * self.eta

            # calculate gradients
            weight_gradients = np.dot(activations[i-1].T, delta)
            bias_gradients = np.sum(delta, axis=0, keepdims=True)

            # update weights and bias
            self.weights[i] -= weight_gradients
            self.bias[i] -= bias_gradients

    # training
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
                print(X_batch.shape, y_batch.shape)
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
    


    



    