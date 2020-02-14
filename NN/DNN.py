from NN.utils import initialize_parameters_deep, L_model_forward, L_model_backward, update_parameters, load_data, \
    compute_cost_with_regulirazation, initialize_adam, initialize_velocity, random_mini_batches,\
    update_parameters_with_momentum, update_parameters_with_adam
from helper_funcs import *
import scipy.io as sio
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import math
import time


# Digits
dataset = sio.loadmat('C:/Users/u325539/Desktop/ML/proj/ML_regs/log_reg/ex3data1.mat', squeeze_me=True)

# weights = sio.loadmat('log_reg/ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X']  # [5000, 400]
y_data_orig = dataset['y']  # [5000, 1]

x_train_dig = X_data_orig.T
y_train_dig = y_data_orig.reshape(5000, -1)
ohe = OneHotEncoder(categories='auto')
y_train_dig = ohe.fit_transform(y_train_dig).toarray()
y_train_dig = y_train_dig.T

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print ("train_y's shape: " + str(train_y.shape))
print ("test_y's shape: " + str(test_y.shape))

layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
layers_dims_dig = [400, 64, 32, 32, 10]

print ("train_x_dig's shape: " + str(x_train_dig.shape))
print ("test_x_dig's shape: " + str(y_train_dig.shape))


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000,
                  print_cost=False, lambd=0, mini_batch_size=64, beta = 0.9,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer="gd"):

    np.random.seed(1)
    costs = []  # keep track of cost
    seed = 10
    m = X.shape[1]

    # Parameter initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Optimizer
    if optimizer == "gd":
        pass
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Gradient Descent
    for i in range(0, num_iterations):

        # Mini_batch loop
        seed = seed+1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # mini-batch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Cost
            if lambd == 0:
                cost_total += compute_cost(AL, minibatch_Y)
            else:
                cost_total = compute_cost_with_regulirazation(AL, minibatch_Y, caches, lambd)

            # Back propagation
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)

            # Update grads
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        cost_avg = cost_total / m

        # Print the cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


start = time.clock()
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True, lambd=0.001)
# parameters_2 = L_layer_model(x_train_dig, y_train_dig, layers_dims_dig, num_iterations = 2500, print_cost = True, lambd=0.1)

end = time.clock()
print("Elapsed time: ", end-start)
