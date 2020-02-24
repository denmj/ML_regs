from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
# from t_flow.python.framework import ops
from t_flow.tf_utils import load_dataset


print(tf.__version__)
print(tf.executing_eagerly())

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test,  y_test, verbose=2)

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39


print(y_hat, y)
loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

"""
 When init is run later (session.run(init))
 Session runs are replaced by eager execution in tf version 2.0
"""

# init = tf.initialize_all_variables()
#                                                   # the loss variable will be initialized and ready to be computed
# with tf.Session() as session:                    # Create a session and print the output
#     session.run(init)                            # Initializes the variables
#     print(session.run(loss))

