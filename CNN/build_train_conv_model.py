import tensorflow as tf
from keras.models import Sequential
import keras
from keras import layers
from t_flow.tf_utils import *
import os


def create_model():
    model = Sequential()
    model.add(layers.Conv2D(8, (4, 4), strides=(1, 1), padding="SAME", activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((8, 8), strides=(8, 8), padding="SAME"))
    model.add(layers.Conv2D(16, (2, 2), activation='relu', strides=(1, 1), padding="SAME"))
    model.add(layers.MaxPooling2D((4, 4), strides=(4, 4), padding="SAME" ))

    model.add(layers.Flatten())
    model.add(layers.Dense(6))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model


if __name__ == '__main__':

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)
    print(tf.__version__)
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
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

    model = create_model()
    model.summary()

    model.fit(X_train, Y_train, epochs=150, batch_size=64,
                        validation_data=(X_test, Y_test),
                        callbacks=[cp_callback])


