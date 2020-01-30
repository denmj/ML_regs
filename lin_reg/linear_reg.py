import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper_funcs import *

#pre precessing
from sklearn import preprocessing as prep

# ml models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('ex1data1.txt', sep=",", header=None)


def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


def h(x, theta):
    return x.dot(theta)


dataset = df.to_numpy()

X = dataset[:, :1]
X_b = np.c_[np.ones([len(X), 1]), X]
X_b_n = np.c_[np.ones([len(X), 1]), normalize(X)]
y = dataset[:, 1:2]
t = np.zeros([2, 1])

# Model

# Test/Train split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)


lm = LinearRegression()
model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
print(model.score(X_test, y_test))


def cost_and_grad(y, x, theta, l=0):
    m = len(y)
    cost_reg_term = (l / (2 * m)) * sum(theta[1:, :] ** 2)
    grad_reg_term = (l/m) * sum(theta[1:, :])
    cost = 1 / (2 * m) * sum(
        (h(x, theta) - y) ** 2) + cost_reg_term
    grad = (1/m) * x.T.dot((h(x, theta) - y)) + grad_reg_term
    return cost, grad


def grad_descent_n(y, x, theta, alpha, iter):
    t = theta
    m = len(y)
    cost_history = []
    theta_history = []
    for i in range(iter):
        delta = 1 / m * (
            x.T.dot((h(x, t) - y)))  # 1/2*m  * 2x(mx - b) partial derivative of cost func with respect to theta
        t = t - alpha * delta
        c, g= cost_and_grad(y, x, t)
        cost_history.append(c)
        theta_history.append(t)

    return t, c, cost_history, theta_history


nt, nc, ch, th = grad_descent_n(y, X_b, t, 0.01, 1500)
ntn, ncn, chn, thn = grad_descent_n(y, X_b_n, t, 0.01, 500)

h0 = h(X_b, nt)
h_norm = h(X_b_n, ntn)


def plotData(y, X, h):
    plt.scatter(X[:, 1:], y, color='red')
    plt.plot(X[:, 1:], h)
    plt.title("Linear Regression")
    plt.xlabel('X variable')
    plt.ylabel('Y variable')
    plt.legend({'Training Data', 'Linear Regression'})
    plt.show()


def plotCost(c_h):
    plt.plot(c_h)
    plt.title("Error rate")
    plt.xlabel('Training iter')
    plt.ylabel('MSE')
    plt.show()


def mpe(x, y, new_theta):
    n = len(y)
    return 100 - (100 / n) * abs(sum((x.dot(new_theta) - y) / x.dot(new_theta)))


# print('Theta parameter for not normalized data: ', nt[0], nt[1])
# print('Minimized cost for not normalized data: ', nc)
# print("Accuracy: ", mpe(X_b, y, nt), "%")
#
#
# # print(c_hist)
# print(50 * '@')
# print('Theta parameter for normalized data: ', ntn[0], ntn[1])
# print('Minimized cost for normalized data: ', ncn)
# print("Accuracy: ", mpe(X_b_n, y, ntn), "%")
#
# plotData(y, X_b, h0)
# plotCost(ch)
# plotData(y, X_b_n, h_norm)
# plotCost(chn)
