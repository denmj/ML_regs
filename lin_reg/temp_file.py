from helper_funcs import *
import scipy.io as sio


# Digits
dataset = sio.loadmat('C:/Users/u325539/Desktop/ML/proj/ML_regs/log_reg/ex3data1.mat', squeeze_me=True)

# weights = sio.loadmat('log_reg/ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X'] # [400, m]
y_data_orig = dataset['y']
print(X_data_orig.shape)

classes = 10

INPUT_LAYER = 400
HIDDEN_L_1 = 64
HIDDEN_L_2 = 32
OUTPUT_LAYER = 10
l = [X_data_orig.shape[1], classes, 1]
params  = parameters_initialization(l)

print(params.keys())

print("shape of w1: {}".format(params['W1'].shape))

print("shape of w1_1: {}".format(params['W2'].shape))

# print("shape of w3: {}".format(params['W3'].shape))
# print("shape of b3: {}".format(params['b3'].shape))
print(len(params) // 2)


for layer in range (1, (len(params) //2) + 1 ):
    print("Layer number {} parameters".format(layer))

    print("Layer shape is {} ".format(params['W' + str(layer)].shape))
    print(params['W' + str(layer)])
    print(params['b' + str(layer)])

