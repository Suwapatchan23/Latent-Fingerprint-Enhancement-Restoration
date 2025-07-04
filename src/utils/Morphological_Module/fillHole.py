import cv2 as cv
import numpy as np

def fillHoles(input_img):
    
    buffer_img = np.zeros((input_img.shape[0]+2, input_img.shape[1]+2) , np.uint8)
    
    _, flood_img, _, _ = cv.floodFill(input_img.copy(), buffer_img, (0, 0), 1)
    
    hole_img = np.logical_not(flood_img)
    
    output_img = np.logical_or(input_img, hole_img)
    output_img = output_img.astype(bool)
    
    return output_img