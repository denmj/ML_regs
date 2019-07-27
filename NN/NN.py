import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.io as sio
from PIL import Image

dataset = sio.loadmat('ex4data1.mat', squeeze_me=True)
weights = sio.loadmat('ex4weights.mat', squeeze_me=True)

# Numpy for matrix operations
X = dataset['X']
y = dataset['y']
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
# Pandas for data info
dfX = pd.DataFrame(data=X, index=X[:, 0], columns=X[0, :])
dfy = pd.DataFrame(data=y)

print(Theta1.shape)
print(Theta2.shape)
print(dfX.info())
print(dfy.info())

img1 = np.reshape(X[1, :], (20, 20))
img2 = np.reshape(X[2, :], (20, 20))
img_pair = np.hstack((img1, img2))
# check image in data set


# displays digits in a row , orientation is not right still
def dispData(num_of_digits):
    img_arr_i = np.reshape(X[1, :], (20, 20))
    for i in range(num_of_digits):
        temp_arr_i = np.reshape(X[np.random.randint(0, 4999), :], (20, 20))
        img_arr_i = np.hstack((img_arr_i, temp_arr_i))

    im = Image.fromarray(img_arr_i * 255)
    im.show()


dispData(20)