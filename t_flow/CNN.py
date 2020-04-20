import math
import numpy as np
import scipy
from scipy import ndimage

from t_flow import tf_utils
import tensorflow as tf
import matplotlib.pyplot as plt
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)


fig = plt.figure(figsize=(10,10))
plt.title("Label of y: {}".format(Y_train_orig[0][0]))
plt.imshow(X_train_orig[0])
plt.show()


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6).T


X, Y = tf_utils.create_placeholders(64,64,3,6)

params = tf_utils.initialize_parameters()

print(params)


_, _, parameters = tf_utils.model(X_train, Y_train, X_test, Y_test)
