import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def add_salt_and_pepper(gb, prob):
    '''Adds "Salt & Pepper" noise to an image.
    gb: should be one-channel image with pixels in [0, 1] range
    prob: probability (threshold) that controls level of noise'''

    rnd = np.random.rand(gb.shape[0], gb.shape[1])
    noisy = gb.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy


# img = mpimg.imread('lena512.png')
# gray = rgb2gray(img)
# gray_salt_pepper = add_salt_and_pepper(gray, 0.01)
# gray_flated = gray.reshape(-1)[:10000]
# plt.plot(range(len(gray_flated)), gray_flated, label='real Image')
# gray_flated = gray_salt_pepper.reshape(-1)[:10000]
# plt.plot(range(len(gray_flated)), gray_flated, label='salt Image papper')
# plt.legend()
# plt.show()
# plt.imshow(gray, cmap = plt.get_cmap('gray'))
# plt.show()