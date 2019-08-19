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


def predict(vec_theta, x, y):
    t1 = np.reshape(vec_theta[0:10025], (25, 401))
    t2 = np.reshape(vec_theta[10025:], (10, 26))
    X1 = np.c_[np.ones([len(x), 1]), x]
    h1 = sigmoid(X1.dot(t1.T))
    X2 = np.c_[np.ones([len(h1), 1]), h1]
    h2 = sigmoid(X2.dot(t2.T))
    pred_vals = np.array([])
    for i in range(len(h2)):
        max_ind = np.argmax(h2[i, :]) + 1  # Adding 1 to match  y-target values where 1=1...10=0
        pred_vals = np.append(pred_vals, max_ind)

    print('The accuracy of the model:\n  {:.1%}'.format(np.mean(pred_vals == y)))


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


t1_u = Theta1.ravel()
t2_u = Theta2.ravel()
Theta_unr = np.concatenate([t1_u, t2_u], axis=0)


#unroll and reshape parameters example.
# A = np.array([[1, 2, 3], [4, 5, 6], [4, 9, 6], [6, 7, 8]])
# B = A.ravel()
# print(A)
# print(B)
# C = np.reshape(B[0:6], (2, 3))
# D = np.reshape(B[6:], (2, 3))
# print(C)
# print(D)


def costfunc(theta_unroll, X ,y, l=0):

    t1 = np.reshape(theta_unroll[0:10025], (25, 401))
    t2 = np.reshape(theta_unroll[10025:], (10, 26))

    epsilon = 1e-5
    m = len(y)
    # turing y from [5000,1] into [5000,10] matrix
    a0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y1 = mtlb.repmat(a0, 5000, 1)
    Y2 = mtlb.repmat(y, 10, 1).T
    Y = np.equal(Y1, Y2).astype(int)

    # Feed forward
    a1 = np.c_[np.ones([len(X), 1]), X]
    z2 = a1.dot(t1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones([len(a2), 1]), a2]
    z3 = a2.dot(t2.T)
    a3 = sigmoid(z3)

    reg_term = (l / (2 * m)) * (sum(sum((t1[1:] ** 2))) + sum(sum((t2[1:] ** 2))))
    J = (1 / m) * sum(sum((-Y * np.log(a3 + epsilon)) - ((1 - Y) * np.log(1 - a3 + epsilon))))
    J = J + reg_term

    # Backpropagation (computing gradient)
    error3 = a3 - Y  # This is a dC/dz for output layer
    der_sig = sigmoid(z2, derivative=True) # add bias 1's to this
    der_sig = np.c_[np.ones([len(der_sig), 1]), der_sig]
    d2 = error3.dot(t2)

    error2 = np.multiply(d2, der_sig) # dC/dz for hidden layer
    error2 = np.delete(error2, 0, 1)

    delta2 = error3.T.dot(a2)
    delta1 = error2.T.dot(a1)

    # print(delta2.shape)
    # print(delta1.shape)

    reg_grad_temp1 = (l / m) * Theta1[1:]
    reg_grad_temp2 = (l / m) * Theta2[1:]

    reg_param_for_grad1 = np.insert(reg_grad_temp1, 0, 0, axis=0)
    reg_param_for_grad2 = np.insert(reg_grad_temp2, 0, 0, axis=0)

    grad1 = delta1 + reg_param_for_grad1
    grad2 = delta2 + reg_param_for_grad2

    t1_unr = grad1.T.ravel()
    t2_unr = grad2.T.ravel()
    t_unr = np.concatenate([t1_unr, t2_unr], axis=0)

    return J, t_unr


# Training
options= {'maxiter': 100}
lambda_ = 1
costFunction = lambda p: costfunc(p, X, y, lambda_)
# Now, costFunction is a function that takes in only one argument
res = op.minimize(costFunction,
                        Theta_unr,
                        jac=True,
                        method='TNC',
                        options=options)

print(res.x)


predict(res.x, X, y)
