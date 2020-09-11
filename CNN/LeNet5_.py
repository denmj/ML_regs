import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


# Classic LeNet5
input_shape = (28, 28, 1)
num_classes = 10

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(6, kernel_size=(3, 3), activation='relu')(inputs)
x = layers.AveragePooling2D()(x)
x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = layers.AveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(120, activation='relu')(x)
x = layers.Dense(84, activation='relu')(x)
out = layers.Dense(num_classes, activation='softmax')(x)


model = keras.Model(inputs, out)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",  metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='/resources/Machine-Learning-models/CNN_keras/',
        save_freq='epoch')
]

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=2, callbacks= callbacks, validation_split = 0.2)