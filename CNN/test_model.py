from CNN.build_train_conv_model import create_model
from t_flow.tf_utils import *
from PIL import Image
import numpy as np
from matplotlib import pyplot
# Open the image form working directory

# image = Image.open('C:/Users/denis/Desktop/ML/ML_regs/CNN/one_.jpg')
# image = image.resize((64, 64))
#
# image = np.asarray(image)
# image = image.reshape(1, image.shape[0],
#                       image.shape[1],
#                       image.shape[2])
# print(image.shape)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes)
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

checkpoint_path = 'C:/Users/denis/Desktop/ML/ML_regs/CNN/cp.ckpt'
cnn_model = create_model()

cnn_model.load_weights(checkpoint_path)
loss, acc = cnn_model.evaluate(X_test, Y_test)

print(loss, acc)
t = np.asarray(X_test[15])
t = t.reshape(1, 64, 64, 3)
pred = cnn_model.predict(t)

print(np.argmax(pred, 1))
print(Y_test[15])
