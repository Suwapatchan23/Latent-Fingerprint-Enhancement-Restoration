import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.Morphological_Module import fillHoles
from utils.Normalize_Module import normalize, adjustRange
from scipy.signal import convolve2d, windows

def applied_gaussian(mask, kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize the kernel

    filter = convolve2d(mask, kernel, mode="same")
    return filter

input_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\003L8U.bmp"


img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

output = img.copy()


global_mean = int(np.mean(img))


threshold = 40


for y in range(1, img.shape[0] - 1):
    for x in range(1, img.shape[1] - 1):
        patch = img[y-1:y+2, x-1:x+2]
        

        
        if np.all(patch <= threshold):
            output[y-1:y+2, x-1:x+2] =  global_mean # แทนด้วยค่าเฉลี่ย


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(output, cmap='gray'); plt.title("After Patch Replace")
plt.tight_layout(); plt.show()