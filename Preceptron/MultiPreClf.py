"""
    Multilayer Perceptron

"""

import numpy as np

class MLP(object):
    def __init__(self, alpha=0.01, n_iterations=50, input_layer_size=784, hidden_layers_size=[128, 64],
                 n_outputs=10, regularization='l2', lambda_reg=0.01):
        np.random.seed(42)  # For reproducibility
        self.eta = alpha
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.n_iterations = n_iterations
        self.input_layer_size = input_layer_size
        self.hidden_layers_size = hidden_layers_size
        self.n_outputs = n_outputs
        self.weights = []
        self.bias = []
        self.layer_sizes = [self.input_layer_size] + self.hidden_layers_size + [self.n_outputs]

    def init_weights(self, he=True):
        for i in range(len(self.layer_sizes) - 1):
            if he:
                he_std_dev = np.sqrt(2 / self.layer_sizes[i])
                self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * he_std_dev)
            else:
                self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) - 0.5)
                self.bias.append(np.random.rand(self.layer_sizes[i + 1], 1) - 0.5)

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            linear_output = self.weights[i].dot(activations[i]) + self.bias[i]
            activation_output = self.relu(linear_output)
            activations.append(activation_output)
        last_linear_output = self.weights[-1].dot(activations[-1]) + self.bias[-1]
        if self.n_outputs == 1:
            last_activation_output = self.sigmoid(last_linear_output)
        else:
            last_activation_output = self.softmax(last_linear_output)
        activations.append(last_activation_output)
        return activations

    def backpropagation(self, X, y, activations):

        n_samples = X.shape[0]

        w_grads = [np.zeros_like(w) for w in self.weights]
        b_grads = [np.zeros_like(b) for b in self.bias]

        if self.n_outputs == 1:
            error = activations[-1] - y  
        else:
            error = (activations[-1] - y) / activations[-1] * (1 - activations[-1])
        
        for layer in reversed(range(len(self.weights))):
            w_grads[layer] = np.dot(activations[layer].T, error) / n_samples
            b_grads[layer] = np.sum(error, axis=0, keepdims=True) / n_samples
            if layer > 0:
                error = np.dot(error, self.weights[layer].T) * self.relu_prime(activations[layer])

        if self.regularization == 'l2':
            w_grads[layer] += (self.lambda_reg / n_samples) * self.weights[layer]
        elif self.regularization == 'l1':
            w_grads[layer] += (self.lambda_reg / n_samples) * np.sign(self.weights[layer])
        
        # Update weights and biases
        for layer in range(len(self.weights)):
            self.weights[layer] -= self.eta * w_grads[layer]
            self.bias[layer] -= self.eta * b_grads[layer]

        return error

    def back_propagation(self, X, y, activations):

        error = None
        for layer in range(1, len(self.weights) + 1):
            if layer == len(self.weights):
                error = activations[-1] - y
            else:
                if error is not None:
                    error = np.dot(error, self.weights[-layer + 1].T) * self.relu_prime(activations[-layer])
                else:
                    # Handle case where error is not yet defined
                    error = np.zeros_like(activations[-layer])

            if self.regularization == 'l2':
                reg_penalty = self.lambda_reg * self.weights[-layer]
            elif self.regularization == 'l1':
                reg_penalty = self.lambda_reg * np.sign(self.weights[-layer])
            else:
                reg_penalty = 0
            delta = error * self.eta
            self.weights[-layer] -= (np.dot(activations[-layer - 1].T, delta) + reg_penalty)
            self.bias[-layer] -= np.sum(delta, axis=0, keepdims=True)

    def train(self, X, y, X_val=None, y_val=None, batch_size=20, verbose=True):
        
        self.init_weights()

        training_loss = []
        validation_loss = []
        for i in range(self.n_iterations):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            batch_losses = []

            for j in range(0, X.shape[0], batch_size):

                X_batch = X_shuffled[j:j + batch_size]
                y_batch = y_shuffled[j:j + batch_size]

                activations = self.forward_propagation(X_batch)
                batch_loss = self.cross_entropy_loss(y_batch, activations[-1])
                batch_losses.append(batch_loss)

                self.backpropagation(X_batch, y_batch, activations)

            epoch_loss = np.mean(batch_losses)
            pred = self.predict(X)
            acc = self.accuracy(np.argmax(y, axis=1), pred)
            if verbose: 
                print(f'Epoch: {i}, Training loss : {epoch_loss}')
                print(f'Epoch: {i}, Training accuracy : {acc}')

            training_loss.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_activations = self.forward_propagation(X_val)
                val_loss = self.cross_entropy_loss(y_val, val_activations[-1])
                validation_loss.append(val_loss)
                if verbose: 
                    print(f'Epoch: {i}, Validation loss: {val_loss}')

        return training_loss, validation_loss

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def relu(self, z):
        return np.maximum(z, 0)

    def relu_prime(self, z):
        return (z > 0).astype(int)

    def softmax(self, z):
        A = np.exp(z) / sum(np.exp(z))
        return A
    
    def cross_entropy_loss(self, y_true, y_pred):
        if y_true.shape[1] == 1:
            return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred, axis=0) / len(y_true)

    def predict(self, X):
        predictions =  self.forward_propagation(X)[-1]
        return np.argmax(predictions, axis=1) if self.n_outputs > 1 else (predictions > 0.5).astype(int)

    def simple_fwd(self,W1, b1, W2, b2, X):
        
        Z1 = W1.dot(X) + b1
        A1 = self.relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def simple_bckprop(self,  Z1, A1, Z2, A2, W1, W2, X, y):
            
        dZ2 = A2 - y
        dW2 = 1 / X.shape[0] * dZ2.dot(A1.T)
        db2 = 1 / X.shape[0] * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.relu_prime(Z1)
        dW1 = 1 / X.shape[0] * dZ1.dot(X.T)
        db1 = 1 / X.shape[0] * np.sum(dZ1)
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, alpha):

        self.weights[0] -= alpha * dW1
        self.bias[0] -= alpha * db1
        self.weights[1] -= alpha * dW2
        self.bias[1] -= alpha * db2

        return self.weights[0], self.bias[0], self.weights[1], self.bias[1]

    def train_two(self, X, y, alpha, iter):

        W1,  W2 = self.weights
        b1, b2 = self.bias

        for i in range(iter):
            Z1, A1, Z2, A2 = self.simple_fwd(W1, b1, W2, b2, X)

            dW1, db1, dW2, db2 = self.simple_bckprop(Z1, A1, Z2, A2, W1, W2, X, y)

            W1, b1, W2, b2 = self.update_params(dW1, db1, dW2, db2, alpha)

            if i % 10 == 0:
                # print(f'Iteration: {i}, Training loss : {self.cross_entropy_loss(y, A2)}')
                print(f'Iteration: {i}, Training accuracy : {self.accuracy(np.argmax(y, axis=1), np.argmax(A2, axis=1))}')

        return W1, b1, W2, b2