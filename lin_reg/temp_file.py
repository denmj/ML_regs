from helper_funcs import *
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder


# Digits
dataset = sio.loadmat('C:/Users/denis/Desktop/ML/ML_regs/log_reg/ex3data1.mat', squeeze_me=True)

# weights = sio.loadmat('log_reg/ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X']  # [5000, 400]
y_data_orig = dataset['y']  # [5000, 1]

X_train = X_data_orig.T
y_train = y_data_orig.T
# y_train = y_train.reshape(1, 5000)

y_train = y_train.reshape(len(y_data_orig), 1)
ohe = OneHotEncoder(categories='auto')
y_train = ohe.fit_transform(y_train).toarray()
y_train = y_train.T


INPUT_LAYER = 400
HIDDEN_L_1 = 64
HIDDEN_L_2 = 32
HIDDEN_L_3 = 32
OUTPUT_LAYER = 10
M = y_data_orig.shape[0]

dims = [INPUT_LAYER, HIDDEN_L_1, HIDDEN_L_2,HIDDEN_L_3, OUTPUT_LAYER]
params = parameters_initialization(dims)

print("Initialized parameters for network: ",  params.keys())
print("X data set: ", X_train.shape)
print("y data set ", y_train.shape)
print("M size: ", M)
for key in params:
    print(key, params[key].shape)
print("@"*20)
A0 = X_train


AL, caches = linear_activation_forward(A0, params)
cost = compute_cost(AL, y_train)
print(cost)
print(AL.shape)
dZ = AL - y_train
