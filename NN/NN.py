import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.io as sio

dataset = sio.loadmat('ex4data1.mat', squeeze_me=True)
weights = sio.loadmat('ex4weights.mat', squeeze_me=True)


# Numpy for matrix operations
X = dataset['X']
y = dataset['y']
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
# Pandas for data info
dfX = pd.DataFrame(data=X, index=X[:, 0], columns=X[0, :])
dfy = pd.DataFrame(data=y)

print(Theta1.shape)
print(Theta2)
print(dfX.info())
print(dfy.info())

