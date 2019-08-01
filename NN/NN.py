import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.io as sio
from PIL import Image


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


dataset = sio.loadmat('ex4data1.mat', squeeze_me=True)
weights = sio.loadmat('ex4weights.mat', squeeze_me=True)

# Numpy for matrix operations
X = dataset['X']  # [5000, 400] matrix
y = dataset['y']  # [5000, 1] matrix
Theta1 = weights['Theta1']  # [25, 401] matrix
Theta2 = weights['Theta2']  # [10, 26] matrix

# Pandas for data info
dfX = pd.DataFrame(data=X, index=X[:, 0], columns=X[0, :])
dfy = pd.DataFrame(data=y)

INPUT_LAYER_SIZE = 400
HIDDEN_LAYER_SIZE = 25
NUM_LABELS = 10

t1_vect = np.reshape(Theta1, (len(Theta1) * len(Theta1[0]), 1))
t2_vect = np.reshape(Theta2, (len(Theta2) * len(Theta2[0]), 1))
params = np.vstack((t1_vect, t2_vect))
NUM_PARAMS = len(params)
print(NUM_PARAMS)

print(Theta1.shape)
print(Theta2.shape)
print(dfX.info())
print(dfy.info())

img1 = np.reshape(X[1, :], (20, 20))
img2 = np.reshape(X[2, :], (20, 20))
img_pair = np.hstack((img1, img2))


# check image in data set


# displays digits in a row , orientation is not right still
def dispData(num_of_digits):
    img_arr_i = np.reshape(X[1, :], (20, 20))
    for i in range(num_of_digits):
        temp_arr_i = np.reshape(X[np.random.randint(0, 4999), :], (20, 20))
        img_arr_i = np.hstack((img_arr_i, temp_arr_i))

    im = Image.fromarray(img_arr_i * 255)
    im.show()


# Feed forward
def feed_forward(X, t1, t2):
    a1 = np.c_[np.ones([len(X), 1]), X]
    print(a1.shape)
    z2 = a1.dot(t1.T)
    print(z2.shape)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones([len(a2), 1]), a2]
    print(a2.shape)
    z3 = a2.dot(t2.T)
    print(z3.shape)
    a3 = sigmoid(z3)
    print(a3.shape)


def backpropogation():
    pass
