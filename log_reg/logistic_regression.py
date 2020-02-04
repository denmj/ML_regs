import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from helper_funcs import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from log_reg.LogRegr import LogisticRegression_manual

#sklearn model
from sklearn.linear_model import LogisticRegressionCV



# Digits
dataset = sio.loadmat('ex3data1.mat', squeeze_me=True)
weights = sio.loadmat('ex3weights.mat', squeeze_me=True)

X_data_orig = dataset['X']
y_data_orig = dataset['y']

# show_images(random_pick(X_data_orig, 20), 5, 4)


# Reshape and split data
X_reshaped = X_data_orig.reshape(X_data_orig.shape[0], 20, 20)
X_train, X_test, y_train, y_test = train_test_split(X_data_orig, y_data_orig, test_size=0.33, random_state=42)

learn_rate = 0.1
epochs = 3000
classes = 10

# clf = LogisticRegressionCV(cv = 5, max_iter=1000, random_state=0).fit(X_train, y_train)
# print(clf.score(X_train, y_train))


w, b = w_b_initialization(X_train.shape[1], classes)

dw, db, cost = cost_grad_log_reg(w, b, X_train, y_train, Multicalss=True)
costs, w1, b1 = optimize(w, b, X_train, y_train, epochs, learn_rate,  mult=True)

# ----------------------------
A = predict(w1, b1, X_train)
A1 = np.argmax(A, axis=1)
# Train
print(A1[0:20])
print(y_train[0:20]-1)

A_test = predict(w1, b1, X_test)
A1_test = np.argmax(A_test, axis=1)
# Test
print(A1_test[0:20])
print(y_test[0:20]-1)

model = LogisticRegression_manual(X_train, y_train, epochs, learn_rate, multiclass=True, classes=10)
model.fit()