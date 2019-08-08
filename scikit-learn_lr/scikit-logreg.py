import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.io as sio
from sklearn.linear_model import LogisticRegression



def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


# multi class data set
dataset = sio.loadmat('ex3data1.mat', squeeze_me=True)
weights = sio.loadmat('ex3weights.mat', squeeze_me=True)
X = dataset['X']  # [5000, 400] matrix
X = normalize(X)
X_b = np.c_[np.ones([len(X), 1]), X]  # [5000, 401] - adding bias
y = dataset['y']  # [5000, 1] matrix
t_m = np.zeros([len(X_b[0]), 1])
print(t_m.shape)

# 1 class data set
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

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000).fit(X, y)
predictions = clf.predict(X)
print(clf.score(X, y))
print(predictions)
