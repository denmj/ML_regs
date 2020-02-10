import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax as sf


#
# Activation functions e ct
#

def linear_func(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    softmax = x_exp / x_sum
    return softmax


def relu(z, derivative=False):
    relu = np.maximum(0, z)
    if derivative:
        z[z<=0] = 0
        z[z>0] = 1
        relu = z
    return relu


def relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_derivative(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


#
# Helper func for models
#

def compute_cost(AL, Y):
    epsilon = 1e-5
    # Y = Y.reshape(len(Y), 1)
    # y_train_reshaped = y_train_reshaped.T
    # ohe = OneHotEncoder(categories='auto')
    # Y = ohe.fit_transform(Y).toarray()
    # Y = Y.T
    m = 5000
    print(Y.shape)
    print(AL.shape)
    cost = np.mean(-1 / m * np.sum(Y.T * np.log(AL.T) + (1 - Y.T) * np.log(1 - AL.T), axis=1, keepdims=True))
    cost = np.squeeze(cost)
    return cost


def cost_grad_log_reg(w, b, X, y, Multicalss=False):
    """
    w - weights, a np array of size ( features, 1)
    b - bias, a scalar

    """
    if not len(X.shape) == 2:
        X_flattened = X.reshape(X.shape[1] * X.shape[2], -1).T
    else:
        X_flattened = X
    m = X_flattened.shape[1]
    print(m)
    if Multicalss:
        # Multi-class

        y_train_reshaped = y.reshape(len(y), 1)
        ohe = OneHotEncoder(categories='auto')
        y_train_reshaped = ohe.fit_transform(y_train_reshaped).toarray()
        print(y_train_reshaped.shape)
        A = softmax(np.dot(X_flattened, w) + b)
        print(A.shape)
        xentropy = -np.sum(y_train_reshaped * np.log(A))
        cost = np.mean(-1 / m * np.sum(y_train_reshaped * np.log(A) + (1 - y_train_reshaped) * np.log(1 - A), axis=1,
                                       keepdims=True))

        dw = 1 / m * np.dot(X_flattened.T, (A - y_train_reshaped))
        db = 1 / m * np.sum(A - y_train_reshaped)
    else:
        # Binary
        A = sigmoid(np.dot(w.T, X_flattened) + b)
        cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A), axis=1, keepdims=True)

        dw = 1 / m * np.dot(X_flattened, (A - y).T)
        db = 1 / m * np.sum(A - y)

    # grads/derivatives
    cost = np.squeeze(cost)

    return dw, db, cost


def optimize(w, b, X, y, n_iterations, alpha, mult=False):
    costs = []

    for epoch in range(n_iterations):

        dw, db, cost = cost_grad_log_reg(w, b, X, y, mult)

        w = w - alpha * dw
        b = b - alpha * db
        if epoch % 100 == 0:
            costs.append(cost)
            print(cost)

    return costs, w, b


def predict(w, b, X):
    if not len(X.shape) == 2:
        X_flattened = X.reshape(X.shape[1] * X.shape[2], -1).T
    else:
        X_flattened = X
    m = X_flattened.shape[1]

    A = softmax(np.dot(X_flattened, w) + b)

    return A


def accuracy_logitic(Y_target, Y_pred):
    accuracy = np.mean(Y_target != Y_pred)
    return accuracy


def parameters_update(parameters, grads, alpha):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - alpha * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - alpha * grads["db" + str(l + 1)]
    return parameters
