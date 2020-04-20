from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from t_flow.tf_utils import load_dataset
from t_flow.tf_utils import convert_to_one_hot
import tensorflow as tf
from tensorflow.python.keras import datasets, layers, models, losses
from tensorflow.python.keras.preprocessing import image
import scipy.ndimage
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_images.shape, train_labels.shape)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


X_train = X_train_orig/255.
X_test = X_test_orig/255
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


print(X_train_orig.shape, Y_train.shape, X_test_orig.shape, Y_test.shape, classes)




# Training set image

# fig = plt.figure(figsize=(10,10))
# plt.title("Label of y: {}".format(Y_train_orig[0][0]))
# plt.imshow(X_train_orig[0])
# plt.show()

# Custom image

img_path = 'C:/Users/u325539/Desktop/ML/proj/ML_regs/t_flow/image_full.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
print('Input image shape:', x.shape)
my_image = plt.imread(img_path)
plt.imshow(my_image)
plt.show()


# Model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6))
model.summary()



model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
#
history = model.fit(X_train, Y_train, epochs=10, batch_size=16,
                    validation_data=(X_test, Y_test))


prediction = model.predict_classes(x)
print(prediction)


ls = [1,3,4,5,6]
l = [x for x in ls if x%2]

print(l)