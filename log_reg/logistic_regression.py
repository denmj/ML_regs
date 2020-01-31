import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from helper_funcs import *
from sklearn.model_selection import train_test_split

# Digits
dataset = sio.loadmat('ex3data1.mat', squeeze_me=True)
weights = sio.loadmat('ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X']
y_data_orig = dataset['y']

# show_images(random_pick(X_data_orig, 20), 5, 4)


# Reshape and split data
X_reshaped = X_data_orig.reshape(X_data_orig.shape[0], 20, 20)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_data_orig, test_size=0.33, random_state=10)


w, b = w_b_initialization(X_train.shape[1] * X_train.shape[2])

learn_rate = 0.0005
epochs = 300

X_flattened = X_train.reshape(X_train.shape[1] * X_train.shape[2], -1).T
print(X_flattened.shape)
print(w.shape)

z = np.dot(X_flattened, w) + b
print(z.shape)
A = softmax(z)

print(A[0])
