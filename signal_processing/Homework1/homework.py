import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn, idct
import cv2
import os

def compress_block(block: np.ndarray):
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 28, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]]
    
    block_dctn = dctn(block)
    block_dctn_jpeg = Q_jpeg*np.round(block_dctn/Q_jpeg)

    # Decoding
    block_jpeg = idctn(block_dctn_jpeg)

    return block_jpeg

image = datasets.ascent()
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
image_jpeg = image.copy()

h, w = image.shape[0], image.shape[1]
block_size = 8
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = image_jpeg[i:i + block_size, j:j+block_size]
        block = compress_block(block)

fig, axs = plt.subplots(2)
axs[0].imshow(image, label="Intial")
axs[1].imshow(image_jpeg, label="jpeg")
plt.show()