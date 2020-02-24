import math
import numpy as np

from t_flow import tf_utils
import tensorflow as tf

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)


X, Y = tf_utils.create_placeholders(64,64,3,6)

params = tf_utils.initialize_parameters()

print(params)