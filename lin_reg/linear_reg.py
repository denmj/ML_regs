import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

df = pd.read_csv('ex1data1.txt', sep=",", header=None)


def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


dfX = df.iloc[:, :1]
dfy = df.iloc[:, 1:2]
dfX_norm = normalize(dfX)
dfX.insert(loc=0, column=2, value=1)
dfX.columns = range(df.shape[1])
dfX_norm.insert(loc=0, column=2, value=1)
dfX_norm.columns = range(df.shape[1])
theta = pd.Series([0, 0])


def h(x, theta):
    return x.dot(theta)


def cost(y, x, theta):
    m = len(y)
    return 1 / (2 * m) * sum((h(x, theta) - y[1]) ** 2)


# h_temp = h(dfX, theta)
# d = 1/len(dfy)* (dfX.T.dot((h_temp - dfy[2])))


def grad_descent(y, x, theta, alpha, iter):
    t = theta
    m = len(y)
    cost_history = []
    theta_history = []
    for i in range(iter):
        delta = 1 / m * (x.T.dot((h(x, t) - y[1])))  # 1/2*m  * 2x(mx - b) partial derivative of cost func with respect to theta
        t = t - alpha * delta
        c = cost(y, x, t)
        cost_history.append(c)
        theta_history.append(t)

    return t, c, cost_history, theta_history


new_theta, new_cost, c_hist, t_hist = grad_descent(dfy, dfX, theta, 0.01, 1500)
new_theta_n, new_cost_n, c_hist_n, t_hist_n = grad_descent(dfy, dfX_norm, theta, 0.01, 500)


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


g = h(dfX, new_theta)
h = h(dfX_norm, new_theta_n)


def plotData(y, X, h):
    plt.scatter(X[1], y, color='red')
    plt.plot(X[1], h)
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
    return 100-(100/n) * abs(sum((x.dot(new_theta) - y[1]) / x.dot(new_theta)))


print('Theta parameter for not normalized data: ', new_theta[0], new_theta[1])
print('Minimized cost for not normalized data: ', new_cost)
print("Accuracy: ", mpe(dfX, dfy, new_theta), "%")

# print(c_hist)

print(50*'@')
print('Theta parameter for normalized data: ', new_theta_n[0], new_theta_n[1])
print('Minimized cost for normalized data: ', new_cost_n)
print("Accuracy: ", mpe(dfX_norm, dfy, new_theta_n), "%")
plotData(dfy, dfX, g)
plotCost(c_hist)
plotData(dfy, dfX_norm, h)
plotCost(c_hist_n)


