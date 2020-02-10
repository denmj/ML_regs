import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax as sf


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
    softmax = x_exp / x_sum
    return softmax


def relu(x, derivative=False):
    relu = np.maximum(0, x)
    if derivative:
        x[x<=0] = 0
        x[x>0] = 1
        relu = x
    return relu


#
# Helper func for models
#
def parameters_initialization(layers_dim, w_values=None):
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
        activation_cache = Z
        A = sigmoid(Z)
    elif activation == "relu":
        Z, linear_func_cache = linear_func(a, W, b)
        activation_cache = Z
        A = relu(Z)
    elif activation == "softmax":
        Z, linear_func_cache = linear_func(a, W, b)
        activation_cache = Z
        A = sf(Z)
    cache = (linear_func_cache, activation_cache)
    return A, cache


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, Y, cache, activation=None):
    linear_cache, activation_cache = cache
    print(dA.shape)
    print(Y.shape)
    if activation == "relu":
        dZ = dA * relu(activation_cache, derivative=True)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    if activation == "sigmoid":
        dZ = dA * sigmoid(activation_cache, derivative=True)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:

        dZ = dA - Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def linear_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    # Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, Y, current_cache)

    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], Y, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def linear_activation_forward(X, params):
    A = X
    caches = []
    L = len(params) // 2
    for l in range(1, L):
        A_previous = A
        A, cache = linear_activation(A_previous, params["W" + str(l)], params["b" + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation(A, params["W" + str(L)], params["b" + str(L)], "softmax")

    caches.append(cache)
    return AL, caches


def compute_cost(Y_hat, Y):
    # y_train_reshaped = Y.reshape(len(Y), 1)
    # ohe = OneHotEncoder(categories='auto')
    # y_train_reshaped = ohe.fit_transform(y_train_reshaped).toarray()
    # y_train_reshaped = y_train_reshaped.T
    m = 5000
    print(Y_hat.shape)
    print(Y.shape)
    cost = np.mean(-1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat), axis=1, keepdims=True))

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


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009

    np.random.seed(1)
    costs = []
    parameters = parameters_initialization(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = linear_activation_forward(X.T, parameters)
        cost = compute_cost(AL, Y)
        grads = linear_model_backward(AL, Y, caches)

        parameters = parameters_update(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters