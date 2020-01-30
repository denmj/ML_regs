import numpy as np
import matplotlib.pyplot as plt


# Data set
def random_pick(X_data, rand_size):
    inx = np.random.randint(X_data.shape[0], size=rand_size)
    rand_data_set = X_data[inx, :]
    return rand_data_set


# Images
def show_images(X_data, cols, rows, cmap = None):

    pic_pix_size = int(np.sqrt(X_data.shape[1]))

    X_reshaped = X_data.reshape(X_data.shape[0], pic_pix_size, pic_pix_size, order='F')
    fig = plt.figure(figsize=(20, 20))

    for image_num in range(X_reshaped.shape[0]):
        fig.add_subplot(rows, cols, image_num + 1)
        plt.imshow(X_reshaped[image_num], cmap=cmap)
    plt.show()


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp/x_sum


def w_b_initialization(size):
    w = np.zeros([size, 1])
    b = 0
    return w, b


def cost_grad_prop(w, b, X, y):
    pass
