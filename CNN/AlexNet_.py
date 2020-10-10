from CNN.BaseModel import BaseModel

from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os


class AlexNetModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super(AlexNetModel, self).__init__()
        self.build_model(input_shape, num_classes)
        self.hist = []

    def build_model(self, input_shape, num_classes):

        inputs = keras.Input(shape=input_shape)
        # CON1 - MAXPOOL1- NORM1
        x = layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x = layers.BatchNormalization()(x)

        # CONV2 - MAXPOOL2 - NORM2

        x = layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x = layers.BatchNormalization()(x)

        # CONV3 - CONV4 - CONV5 - MAXPOOL3

        x = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
        x = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
        x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        # FC6 - FC7 - FC8
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs, callbacks, batch_size = 128, plot = False):
        self.hist = self.model.fit_generator(x_train, epochs=epochs, callbacks=callbacks,
                                             validation_data=y_train)

        if plot:

            plt.plot(self.hist.history['accuracy'])
            plt.plot(self.hist.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(self.hist.history['loss'])
            plt.plot(self.hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def save(self, checkpoint_path):
        pass

    def summary(self):
        self.model.summary()


class ModelLoader(object):

    KLASSES = ['airplane', 'automobile',
              'bird', 'cat',
              'deer', 'dog',
              'frog', 'horse',
              'ship', 'truck']

    def __init__(self, json_model, model_weights):
        print("MNIST model loaded")
        with open(json_model, 'r') as json_file:
            loaded_json_model = json_file.read()
            self.loaded_json_model = model_from_json(loaded_json_model)

        self.loaded_json_model.load_weights(model_weights)

    def predict(self, image):
        self.pred = self.loaded_json_model(image)
        return ModelLoader.KLASSES[np.argmax(self.pred)]