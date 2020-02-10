from helper_funcs import *
import scipy.io as sio


# Digits
dataset = sio.loadmat('C:/Users/denis/Desktop/ML/ML_regs/log_reg/ex3data1.mat', squeeze_me=True)

# weights = sio.loadmat('log_reg/ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X']  # [400, m]
y_data_orig = dataset['y']  # [1, m]

INPUT_LAYER = 400
HIDDEN_L_1 = 64
HIDDEN_L_2 = 32
HIDDEN_L_3 = 32
OUTPUT_LAYER = 10
M = y_data_orig.shape[0]

dims = [INPUT_LAYER, HIDDEN_L_1, HIDDEN_L_2,HIDDEN_L_3, OUTPUT_LAYER]
params = parameters_initialization(dims)

print("Initialized parameters for network: ",  params.keys())
print("X data set: ", X_data_orig.shape)
print("y data setL ", y_data_orig.shape)

for key in params:
    print(key, params[key].shape)

# # Layer 1 linear step to get Z1
# print("Layer 1: ")
# A1, cache1 = linear_activation(X_data_orig.T, params["W1"], params["b1"], "relu")
#
# print("A1: ",  A1.shape)
# print("Saved in cache: A(l-1), W(l), b(l), Z(l)")
#

#
# print("Layer 2:")
# A2, cache2 = linear_activation(A1, params["W2"], params["b2"], "relu")
#
#
# print("A2: ",  A2.shape)
# print("Saved in cache: A(l-1), W(l), b(l), Z(l)")
# for i in cache1:
#     for j in i:
#         print(j.shape)
#
#
# A3, cache3 = linear_activation(A2, params["W3"], params["b3"], "relu")
#
#
# print("A3: ",  A3.shape)
# print("Saved in cache: A(l-1), W(l), b(l), Z(l)")
# for i in cache1:
#     for j in i:
#         print(j.shape)
#
#
# A4, cache4 = linear_activation(A3, params["W4"], params["b4"], "softmax")
#
# print("A4: ",  A4.shape)
# print("Saved in cache: A(l-1), W(l), b(l), Z(l)")
# for i in cache3:
#     print(i.shape)
#
# print(len(params)// 2)
#
# for i in range(1, len(params)//2):
#     print(i)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print("Forward propagation")
AL, caches = linear_activation_forward(X_data_orig.T, params)

grads = linear_model_backward(AL, y_data_orig, caches)
print("A4 shape is : {}".format(AL.shape))

# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#
# c = compute_cost(AL, y_data_orig)
# print("Cost: ", c)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print("Back propagation")
#
# dA_L = - np.divide(y_data_orig, AL) - np.divide((1 - y_data_orig), (1 - AL))
# dZ_L = dA_L - y_data_orig
#
# dW_L = (1/M)*np.dot(dZ_L, A3.T)
#
# print("partial derivative of L wrt A (4th)", dA_L.shape)
# print("partial derivative of L wrt Z (4th)", dZ_L.shape)
# print("partial derivative of L wrt W (4th)", dW_L.shape)
#
# dA_L_3 = np.dot(dZ_L.T, caches[3][1])
#
# g_Z_3 = relu(caches[2][3], derivative=True)
#
# dZ_L_3 = dA_L_3 * g_Z_3.T
#
# print("partial derivative of L wrt A (3rd)", dA_L_3.shape)
# print("partial derivative of L wrt Z (3rd)", dZ_L_3.shape)


