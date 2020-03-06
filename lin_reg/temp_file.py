import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = 'C:/Users/denis/Desktop/ML/ML_regs/lin_reg/test_img.jpg'
img = Image.open(img)
img_arr = np.asarray(img, dtype="int32" )

# plt.imshow(img_arr[:, :, :])
# plt.show()

np.random.seed(1)
x = np.random.randn(4, 3, 3, 3)

print(x[0, :, :, 0])
print(img_arr.shape)
image_arr = img_arr.reshape((1, img_arr.shape[0],
            img_arr.shape[1], img_arr.shape[2]))

# print(image_arr.shape)
# plt.imshow(x[0, :, :, :])
# plt.show()


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   mode='constant', constant_values=(0, 0))
    return X_pad


X_p = zero_pad(x, 1)
Den_p = zero_pad(image_arr, 2)

print(X_p[0, :, :, 0])
# plt.imshow(Den_p[0, :, :, :])
# plt.show()
#
# plt.imshow(X_p[0, :, :, :])
# plt.show()


def conv_step(a_slice_prev, W, b):

    z_step = np.multiply(a_slice_prev, W)
    z_step = np.sum(z_step)
    z_step = z_step + float(b)
    return z_step


np.random.seed(1)
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(3,3,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 1,
               "stride": 2}


def conv_forward(A_prev, W, b, hparameters):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):   # loop over the training examples
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):  # loop over vertical axis of the output volume
            vert_start = h * stride
            vert_end = h * stride + f
            for w in range(n_W):  # loop over horizontal axis of the output volume
                horiz_start = w * stride
                horiz_end = w * stride + f
                for c in range(n_C):
                    # Find the corners of the current "slice" (≈4 lines)

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hparameters)
    return Z, cache

Z, cache_conv = conv_forward(Den_p, W, b, hparameters)

print(Z.shape)

plt.imshow(Z[0, :, :, 6])
plt.show()