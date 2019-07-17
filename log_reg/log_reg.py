import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import random as rnd

df = pd.read_csv('ex2data1.txt', sep=",", header=None)


def sigmoid(z, derivative=False):
    sig = 1. / (1. + np.exp(-z))
    if derivative:
        return sig * (1. - sig)
    return sig


def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


def prob(x, theta):
    return sigmoid(pred(x, theta))


# split training data set to x and y
dfX = df.iloc[:, :2]
dfX_norm = normalize(dfX)

dfX.insert(loc=0, column=2, value=1)
dfX.columns = range(df.shape[1])

dfX_norm.insert(loc=0, column=2, value=1)
dfX_norm.columns = range(df.shape[1])
nx = dfX.to_numpy()

dfy = df.iloc[:, 2:3]
dfy.columns = range(dfy.shape[1])
ny = dfy.to_numpy()

t1 = pd.Series([0, 0, 0])
# Init thetas this way for high number of parameters
theta = pd.DataFrame(np.zeros((3, 1)))
nt = theta.to_numpy()
RegT = 1


def pred(t, x):
    return x.dot(t)


# Cost function that returns cost and gradient of it
def logregcost(theta, x, y, regt=0):
    epsilon = 1e-5
    m = len(y)
    p = sigmoid(pred(theta, x))
    cost = -np.average(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
    cost_reg = cost + (regt / 2 * m) * (np.linalg.norm(theta[1:] ** 2))
    gr = (1 / m) * np.dot(x.T, (p - y))
    return cost_reg, gr


# Using fmin_tnc to find best params method #1
def fit(t, x, y):
    b_t = op.fmin_tnc(logregcost, x0=t, args=(x, y.flatten()))
    return b_t[0]


# Gradient descent method #2
def gradient_descent_lr(x, y, theta, alpha, iter, regt=0):
    t = theta
    c = []
    m = len(y)
    p = sigmoid(pred(t, x))
    cost_history = []
    theta_history = []
    for i in range(iter):
        regterm = (regt / m) * (np.linalg.norm(t[1:]))
        c, g = logregcost(t, x, y, regt)
        delta = alpha * (1 / m) * (x.T.dot((p - y)))
        t = t - delta + regterm
        cost_history.append(c)
        theta_history.append(t)

    return t, c, g,  cost_history, theta_history


def decision_boundary(p):
    return 1 if p >= .5 else 0


def classify(pred):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize()
    return decision_boundary(pred).flatten()


def accuracy(x, y):
    diff = x - y
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


# Data plotting
def plotCost(c_h):
    plt.plot(c_h)
    plt.title("Error rate")
    plt.xlabel('Training iter')
    plt.ylabel('Loss')
    plt.show()


def plotdata(y, x, t_n):

    x_v = pd.Series([dfX.iloc[:, 1].min() - 1, dfX.iloc[:, 2].max() + 1])
    y_v = pd.Series([-t_n[0, 0] + t_n[1, 0] * x_v.iloc[1:2]/ t_n[2, 0],
                     t_n[0, 0] + t_n[1, 0] * x_v.iloc[:1] / t_n[2, 0]])

    pos = df.index[dfy[0] == 1]
    neg = dfy.index[dfy[0] == 0]
    plt.scatter(x.iloc[pos, 1], x.iloc[pos, 2], color='red')
    plt.scatter(x.iloc[neg, 1], x.iloc[neg, 2], color='blue')
    plt.plot(x_v, y_v)

    plt.title("Log Regression")
    plt.xlabel('Test Score 1')
    plt.ylabel('Test Score 2')
    plt.legend({'Admitted', 'Not Admitted', 'Reg Line'})
    plt.show()


t, cc, gg,  cost_h, theta_h = gradient_descent_lr(dfX, dfy, theta, 0.000001, 1)
t_fm = fit(nt, nx, ny).reshape(3,1)
cc_fm, gg_fm = logregcost(t_fm, nx, ny)

# print(sigmoid(pred(t_fm, nx)))
t_n, cc_n, gg_n, cost_h_n, theta_h_n = gradient_descent_lr(dfX_norm, dfy, theta, 0.01, 1)
t_n_r, cc_n_r, gg_n_r, cost_h_n_r, theta_h_n_r = gradient_descent_lr(dfX_norm, dfy, theta, 0.00001, 1, RegT)


# plotCost(cost_h)
# plotCost(cost_h_n)
# plotCost(cost_h_n_r)
print(70 * '-')

print('Theta parameters from fmin_tnc optimizer:\n ', t_fm)
print('Minimized cost from fmin_tnc optimizer:\n ', cc_fm)
print(70 * '-')
print('Theta parameter for not normalized data:\n ', t)
print('Minimized cost for not normalized data:\n ', cc)
print('History of cost: ', cost_h)
print(50 * '-')
print('Theta parameter for  normalized data:\n ', t_n)
print('Minimized cost for  normalized data:\n ', cc_n)
print('History of cost: ', cost_h_n)
print(50*'-')
print('Theta parameter for  normalized data:\n ', t_n_r)
print('Minimized cost for  normalized data:\n ', cc_n_r)
print('History of cost: ', cost_h_n_r)
#
# plotdata(dfy, dfX, t_n)
plotdata(dfy, dfX, t_fm)
