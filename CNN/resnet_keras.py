import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from t_flow.tf_utils import *

import keras
from keras.models import load_model
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint


def res_net_model():
    img_height, img_width = 64, 64
    num_classes = 6
    # If imagenet weights are being loaded,
    # input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model



if __name__ == '__main__':

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    # Y_train = Y_train_orig.T
    # Y_test = Y_test_orig.T

    print(X_train.shape)
    print(Y_train.shape)

    # save model weights
    checkpoint_path = 'C:/Users/denis/Desktop/ML/ML_regs/CNN/resnet_keras.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model = res_net_model()
    model.summary()

    model.fit(X_train, Y_train, epochs=150, batch_size=64,
              validation_data=(X_test, Y_test),
              callbacks=[cp_callback])
