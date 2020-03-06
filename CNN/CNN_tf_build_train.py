import tensorflow as tf
from keras.models import Sequential
import keras
from keras import layers
from t_flow.tf_utils import *
import os

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)
print(tf.__version__)
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print(X_train.shape)
print(Y_train.shape)

# save model weights
checkpoint_path = 'C:/Users/denis/Desktop/ML/ML_regs/CNN/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

def create_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

history = model.fit(X_train, Y_train, epochs=20, batch_size=128,
                    validation_data=(X_test, Y_test),
                    callbacks=[cp_callback])


