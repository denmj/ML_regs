import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.io as sio
from sklearn.preprocessing import PolynomialFeatures


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
    # theta = theta.reshape(3, 1)
    return prob(x, theta)


def prob(x, theta):
    return sigmoid(pred(theta, x))


def accuracy(x, y, theta, threshhold=0.5, num_l=2):
    if num_l == 2:
        predicted_classes = (predict(x, theta) >= threshhold).astype(int)
        # predicted_classes = predicted_classes.flatten()
        acc = np.mean(predicted_classes == y)
    else:
        pass
    return acc


# multi class data set
dataset = sio.loadmat('ex3data1.mat', squeeze_me=True)
weights = sio.loadmat('ex3weights.mat', squeeze_me=True)
X = dataset['X']  # [5000, 400] matrix
X = normalize(X)
X_b = np.c_[np.ones([len(X), 1]), X]  # [5000, 401] - adding bias
y = dataset['y']  # [5000, 1] matrix
t_m = np.zeros([len(X_b[0]), 1]) # [401, 1]

# 2 class data set
df = pd.read_csv('ex2data1.txt', sep=",", header=None)

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

# Another set of data (Polynomial and Use of regularization)
df_2 = pd.read_csv('ex2data2.txt', sep=",", header=None)
X_2 = df_2[[0, 1]].to_numpy()
X_2 = normalize(X_2)
y_2 = df_2[[2]].to_numpy()
X_b_2 = np.c_[np.ones([len(X_2), 1]), X_2]

poly = PolynomialFeatures(degree=2)
X_2_poly = poly.fit_transform(X_2)
t_poly = np.zeros([len(X_2_poly[0]), 1])



def pred(t, x):
    return x.dot(t)


# Cost function that returns cost and gradient of it
def logregcost(theta, x, y, regt=0):
    # print(theta)
    epsilon = 1e-5
    m = len(y)
    p = sigmoid(pred(theta, x))
    reg_param_for_cost = sum((regt / (2 * m)) * (theta[1:] ** 2))
    reg_grad_temp = (3 / m) * theta[1:]
    reg_param_for_grad = np.insert(reg_grad_temp, 0, 0, axis=0)
    cost = (1 / m) * sum((-y * np.log(p + epsilon)) - ((1 - y) * np.log(1 - p + epsilon)))
    cost_reg = cost + reg_param_for_cost
    grad = (1 / m) * np.dot(x.T, (p - y))
    grad_reg = grad + reg_param_for_grad
    return cost_reg, grad_reg


# Using fmin_tnc to find best params method #1
def fit(t, x, y, num_labels=2, reg = 0):

    if num_labels == 2:
        solution = op.fmin_tnc(logregcost, x0=t, args=(x, y.flatten(), reg))
        thetas = solution[0]
    else:
        thetas = np.array([])
        for i in range(1, num_labels + 1):
            y_1 = np.array([(y == i).astype(int)]).T
            solution = op.fmin_tnc(logregcost, x0=t, args=(x, y_1.flatten(), reg))
            thetas = np.concatenate([thetas, solution[0]], axis=0)

    return thetas


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

    return theta.flatten(), c, g, cost_history, theta_history


# Data plotting
def plotCost(c_h):
    plt.plot(c_h)
    plt.title("Error rate")
    plt.xlabel('Training iter')
    plt.ylabel('Loss')
    plt.show()


def plotdata(x, t_n):
    x_v = pd.Series([np.min(x[:, 1]) - 1, np.max(x[:, 2] + 1)])
    y_v = -(t_n[0] + np.dot(t_n[1], x_v)) / t_n[2]
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


# Feed Numpy type / Working on plotting polynomial function as des.boundary
def plottemp (x, y, title, xl, yl):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(x[pos[0], 1], x[pos[0], 2], color='red')
    plt.scatter(x[neg[0], 1], x[neg[0], 2], color='blue')
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend({'Regression line', 'Not Admitted', 'Admitted'})
    plt.show()



plottemp(X_b_2, y_2, "Title", "Micro1", "Micro2")
grad_p, cost_p = logregcost(t_poly, X_2_poly, y_2, 3)

print(grad_p)
print(cost_p)
fmin_theta_poly = fit(t_poly, X_2_poly, y_2, 2, 3)
print(fmin_theta_poly)
print('The accuracy of the model:\n  {:.1%}'.format(accuracy(X_2_poly, y_2.flatten(), fmin_theta_poly)))


# Evaluation of model for 2-class Classification
fmin_theta = fit(nt, nx, ny)
fmin_cost, fmin_grad = logregcost(fmin_theta.reshape(3, 1), nx, ny)
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


# Evaluation of model for multi-class Classification (Picture recognition [20x20])
fmin_theta_m_1 = fit(t_m, X_b, y, 10)
theta_m = fmin_theta_m_1.reshape(10, 401)
pred_values = sigmoid(pred(theta_m.T, X_b))

vals = np.array([])
for i in range(len(pred_values)):
    max_ind = np.argmax(pred_values[i, :])+1  # Adding 1 to match  y-target values where 1=1...10=0
    vals = np.append(vals, max_ind)

print('The accuracy of the model:\n  {:.1%}'.format(np.mean(vals == y)))
print(70 * '-')
