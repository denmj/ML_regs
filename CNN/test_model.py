from CNN.build_train_conv_model import create_model
from t_flow.tf_utils import *
from PIL import Image
import numpy as np
from matplotlib import pyplot
# Open the image form working directory

image = Image.open('C:/Users/denis/Desktop/ML/ML_regs/CNN/five_.jpg')
image = image.resize((64, 64))

image = np.asarray(image)
image = image.reshape(1, image.shape[0],
                      image.shape[1],
                      image.shape[2])
print(image.shape)

checkpoint_path = 'C:/Users/denis/Desktop/ML/ML_regs/CNN/cp.ckpt'
cnn_model = create_model()

cnn_model.load_weights(checkpoint_path)
pred = cnn_model.predict(image)

print(np.argmax(pred, 1))
