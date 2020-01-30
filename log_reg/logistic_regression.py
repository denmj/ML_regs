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

print('X dataset shape is {}'.format(X_data_orig.shape))
print('y dataset shape is {}'.format(y_data_orig.shape))

# show_images(random_pick(X_data_orig, 20), 5, 4)


# Reshape and split data
X_reshaped = X_data_orig.reshape(X_data_orig.shape[0], 20, 20)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_data_orig, test_size=0.33, random_state=10)

print('X_train set shape - {}'.format(X_train.shape))
print('X_test set shape - {}'.format(X_test.shape))
print('y_train set shape - {}'.format(y_train.shape))
print('y_test set shape - {}'.format(y_test.shape))


w, b = w_b_initialization(X_train.shape[1] * X_train.shape[2])
