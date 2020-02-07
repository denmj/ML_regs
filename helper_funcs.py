import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


#
# Data set
#
def random_pick(X_data, rand_size):
    inx = np.random.randint(X_data.shape[0], size=rand_size)
    rand_data_set = X_data[inx, :]
    return rand_data_set


#
# Images
#
def show_images(X_data, cols, rows, cmap=None):
    pic_pix_size = int(np.sqrt(X_data.shape[1]))

    X_reshaped = X_data.reshape(X_data.shape[0], pic_pix_size, pic_pix_size, order='F')
    fig = plt.figure(figsize=(20, 20))

    for image_num in range(X_reshaped.shape[0]):
        fig.add_subplot(rows, cols, image_num + 1)
        plt.imshow(X_reshaped[image_num], cmap=cmap)
    plt.show()


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
    return x_exp / x_sum


def relu(x):
    return np.maximum(0, x)


#
# Helper func for models
#
def parameters_initialization(layers_dim, w_values = None):
    parameters = {}
    if w_values == "zeros":
        for l in range(1, len(layers_dim)):
            parameters['W'+str(l)] = np.zeros(shape=[layers_dim[l], layers_dim[l-1]])
            parameters['b'+str(l)] = np.zeros(shape=[layers_dim[l], 1])
    else:
        for l in range(1, len(layers_dim)):
            parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros(shape=[layers_dim[l], 1])
    return parameters


def linear_activation(a, W, b, activation):
    if activation == "sigmoid":
        Z, linear_func_cache = linear_func(a, W, b)
        sigmoid_func_cache = Z
        A = sigmoid(Z)
    elif activation == "relu":
        Z, linear_func_cache = linear_func(a, W, b)
        sigmoid_func_cache = Z
        A = relu(Z)
    elif activation == "softmax":
        Z, linear_func_cache = linear_func(a, W, b)
        sigmoid_func_cache = Z
        A = softmax(Z)
    cache = list(linear_func_cache)
    cache.append(sigmoid_func_cache)

    return A, cache


def linear_activation_forward(X, params):
    pass


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

    if Multicalss:
        # Multi-class

        y_train_reshaped = y.reshape(len(y), 1)
        ohe = OneHotEncoder(categories='auto')
        y_train_reshaped = ohe.fit_transform(y_train_reshaped).toarray()
        A = softmax(np.dot(X_flattened, w) + b)
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
