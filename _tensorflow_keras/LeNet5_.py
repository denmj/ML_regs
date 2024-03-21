from CNN.BaseModel import BaseModel
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os


class LeNet(BaseModel):
    def __init__(self, input_shape, num_classes):
        super(LeNet, self).__init__()
        self.build_model(input_shape, num_classes)
        self.hist = []

    def build_model(self, input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(6, kernel_size=(3, 3), activation='relu')(inputs)
        x = layers.AveragePooling2D()(x)
        x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
        x = layers.AveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(120, activation='relu')(x)
        x = layers.Dense(84, activation='relu')(x)
        out = layers.Dense(num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, out)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs, callbacks, batch_size=128, plot=False):
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
