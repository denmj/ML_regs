import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

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


def predict(x, theta):
    theta = theta.reshape(3,1)
    return prob(x, theta)


def prob(x, theta):
    return sigmoid(pred(theta, x))


def accuracy(x, y, theta, threshhold=0.5):
    predicted_classes = (predict(x, theta) >= threshhold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == y)
    return accuracy



# split training data set to x and y
dfX = df.iloc[:, :2]
dfX.insert(loc=0, column=2, value=1)
dfX.columns = range(df.shape[1])
nx = dfX.to_numpy()
nxn = normalize(nx)
dfy = df.iloc[:, 2:3]
dfy.columns = range(dfy.shape[1])
ny = dfy.to_numpy()

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
    grad = (1 / m) * np.dot(x.T, (p - y))
    return cost_reg, grad


# Using fmin_tnc to find best params method #1
def fit(t, x, y):
    b_t = op.fmin_tnc(logregcost, x0=t, args=(x, y.flatten()))
    return b_t[0]


# Gradient descent method #2
def gradient_descent_lr(x, y, theta, alpha, iter, regt=0):
    m = len(y)
    p = sigmoid(pred(theta, x))
    cost_history = []
    theta_history = []
    for i in range(iter):
        regterm = (regt / m) * (np.linalg.norm(theta[1:]))
        c, g = logregcost(theta, x, y, regt)
        delta = alpha * (1 / m) * (x.T.dot((p - y)))
        theta = theta - delta + regterm
        cost_history.append(c)
        theta_history.append(theta)

    return theta.flatten(), c, g,  cost_history, theta_history


# Data plotting
def plotCost(c_h):
    plt.plot(c_h)
    plt.title("Error rate")
    plt.xlabel('Training iter')
    plt.ylabel('Loss')
    plt.show()


def plotdata(x, t_n):

    x_v = pd.Series([np.min(x[:,1]) - 1, np.max(x[:,2] + 1)])
    y_v = -(t_n[0] + np.dot(t_n[1], x_v))/t_n[2]
    pos = df.index[dfy[0] == 1]
    neg = dfy.index[dfy[0] == 0]
    plt.scatter(x[pos, 1], x[pos, 2], color='red')
    plt.scatter(x[neg, 1], x[neg, 2], color='blue')
    plt.plot(x_v, y_v)

    plt.title("Log Regression")
    plt.xlabel('Test Score 1')
    plt.ylabel('Test Score 2')
    plt.legend({'Regression line', 'Not Admitted', 'Admitted'})
    plt.show()


fmin_theta = fit(nt, nx, ny)
fmin_cost, fmin_grad = logregcost(fmin_theta.reshape(3,1), nx, ny)
grad_d_theta, cost, grad, cost_hist, theta_hist = gradient_descent_lr(nxn, ny, nt, 0.05, 50)
print(70 * '-')
print('Theta parameters from fmin_tnc optimizer:\n ', fmin_theta)
print('Minimized cost from fmin_tnc optimizer:\n ', fmin_cost)
print('The accuracy of the model:\n  {:.1%}'.format(accuracy(nx, ny.flatten(), fmin_theta)))

print(70 * '-')
print('Theta parameter for  normalized data:\n ', grad_d_theta)
print('Minimized cost for  normalized data:\n ', cost)
print('The accuracy of the model:\n  {:.1%}'.format(accuracy(nxn, ny.flatten(), grad_d_theta)))

plotdata(nxn, grad_d_theta)
plotCost(cost_hist)
plotdata(nx, fmin_theta)
