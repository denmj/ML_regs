from NN.utils import initialize_parameters_deep, L_model_forward, L_model_backward, update_parameters, load_data, \
    compute_cost_with_regulirazation
from helper_funcs import *
import scipy.io as sio
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import math


# Digits
dataset = sio.loadmat('C:/Users/denis/Desktop/ML/ML_regs/log_reg/ex3data1.mat', squeeze_me=True)

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


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, lambd=0):

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameter initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Gradient Descent
    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Cost
        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regulirazation(AL, Y, caches, lambd)

        # Back propagation
        grads = L_model_backward(AL, Y, caches, lambd)

        # Update grads
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#
# par = initialize_parameters_deep(layers_dims)
# AL, caches = L_model_forward(train_x, par)
# print(len(caches))
#
# print(train_x.shape)
# for i in par:
#     print(i, par[i].shape)
#
# sum_of_W = 0
# for numb in range(len(caches)):
#     print(caches[numb][0][1].shape)
#     print(np.sum(np.square(caches[numb][0][1])))
#     sum_of_W = sum_of_W + np.sum(np.square(caches[numb][0][1]))
# print(sum_of_W)

# parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True, lambd=0.1)
# parameters_2 = L_layer_model(x_train_dig, y_train_dig, layers_dims_dig, num_iterations = 2500, print_cost = True, lambd=0.1)

