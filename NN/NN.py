import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as mtlb
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

print(X.shape)
print(y.shape)
print(Theta1.shape)
print(Theta2.shape)

img1 = np.reshape(X[1, :], (20, 20))
img2 = np.reshape(X[2, :], (20, 20))
img_pair = np.hstack((img1, img2))

# check image in data set


# displays digits in a row
def dispData(num_of_digits):
    img_arr_i = np.reshape(X[1, :], (20, 20)).T
    for i in range(num_of_digits):
        temp_arr_i = np.reshape(X[np.random.randint(0, 4999), :], (20, 20)).T
        img_arr_i = np.hstack((img_arr_i, temp_arr_i))

    im = Image.fromarray(img_arr_i * 255)
    im.show()

# dispData(20)


# Feed forward
def costfunc(X, t1, t2, y, l=0):

    epsilon = 1e-5
    m = len(y)
    J = 0
    # turing y from [5000,1] into [5000,10] matrix
    a0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y1 = mtlb.repmat(a0, 5000, 1)
    Y2 = mtlb.repmat(y, 10, 1).T
    Y = np.equal(Y1, Y2).astype(int)
    print("Feed F")
    # Feed forward
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

    reg_term = (l / (2 * m)) * (sum(sum((t1[1:] ** 2))) + sum(sum((t2[1:] ** 2))))
    print(reg_term)
    J = (1 / m) * sum(sum((-Y * np.log(a3 + epsilon)) - ((1 - Y) * np.log(1 - a3 + epsilon))))
    J = J + reg_term

    print("BackP")
    # Backpropagation (computing gradient)
    delta3 = a3 - Y  # This is a dC/dz for output layer

    der_sig = sigmoid(z2, derivative=True) # add bias 1's to this
    der_sig = np.c_[np.ones([len(der_sig), 1]), der_sig]
    d2 = delta3.dot(t2)
    delta2 = np.multiply(d2, der_sig)
    print(delta2.shape)



    reg_term_grad = 0
    grad = 0

    print(J)


costfunc(X, Theta1, Theta2, y, 1)
