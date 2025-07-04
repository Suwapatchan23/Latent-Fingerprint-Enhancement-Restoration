import numpy as np


def normalize(input_img):
    norm_img = (255 * (input_img - input_img.min()) / (input_img.max() - input_img.min())).astype(np.uint8)
    return norm_img