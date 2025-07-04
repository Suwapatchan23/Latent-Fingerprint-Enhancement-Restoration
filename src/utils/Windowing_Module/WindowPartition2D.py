import numpy as np
import cv2 as cv
from utils.Fourier_Module import Fourier2D
from utils.Normalize_Module import adjustRange


class WindowPartition2D:
    
    def __init__(self, input_img, overlap_size, non_overlap_size,
        pad_type=cv.BORDER_CONSTANT, pad_value = None):
        self.__input_img = input_img
        self.__overlap_size = overlap_size
        self.__non_overlap_size = non_overlap_size
        self.__pad_type = pad_type
        
        if pad_value is not None:
            self.__pad_value = pad_value
        else:
            self.__pad_value = input_img.mean()

        # -> Unmatched between "kernel_size" and "stride_size"
        if not (overlap_size%2 == non_overlap_size%2):
            raise TypeError("Unmatched Odd/Even Error: Assign \"kernel_size\" and \"stride_size\"with Odd/Even match.")
        # -> Get image size
        self.__rows = input_img.shape[0]
        self.__cols = input_img.shape[1]
        
        
    def __completeGrid(self):
        
        ### -> Count How many windows/grid gotten
        self.__window_y_count = int(np.ceil(self.__rows/self.__non_overlap_size))
        self.__window_x_count = int(np.ceil(self.__cols/self.__non_overlap_size))
        ### -> Find how many pixels need to be padded.
        rows_grid_extend = (self.__window_y_count * self.__non_overlap_size) - self.__rows
        if rows_grid_extend == 0:
            rows_grid_extend = 1
        # print(rows_grid_extend)
        cols_grid_extend = (self.__window_x_count * self.__non_overlap_size) - self.__cols
        if cols_grid_extend == 0:
            cols_grid_extend = 1
        return rows_grid_extend, cols_grid_extend
    

    def __padding(self):
        
        # -> Find complete grid extend size (Pad to fit grid)
        rows_grid_extend, cols_grid_extend = self.__completeGrid()
        
        ### -> Find window extend size (Pad to expand window)
        border_extend = (self.__overlap_size-self.__non_overlap_size)//2
        window_extend = border_extend * 2
        # -> Find pad size
        self.__pad_rows = rows_grid_extend + window_extend
        self.__pad_cols = cols_grid_extend + window_extend
        ### -> Padding
        self.__input_img_pad = cv.copyMakeBorder(self.__input_img, self.__pad_rows//2,
        self.__pad_rows//2 + self.__pad_rows%2,
        self.__pad_cols//2,
        self.__pad_cols//2 + self.__pad_cols%2,
        self.__pad_type, None, value=self.__pad_value)
        
    def forward(self):
        ### -> Padding Image for Complete Split
        self.__padding()
        ### -> Store All Partition Windows
        self.__window_list = []
        
        for y in range(self.__window_y_count):
            rows_window_list = []
            for x in range(self.__window_x_count):
                y_start = y*self.__non_overlap_size
                y_stop = (y*self.__non_overlap_size)+self.__overlap_size
                x_start = x*self.__non_overlap_size
                x_stop = (x*self.__non_overlap_size)+self.__overlap_size
                partial_img = self.__input_img_pad[y_start:y_stop, x_start:x_stop]
                # -> Each Row
                rows_window_list.append(partial_img)
            # -> Stack Row
            self. __window_list.append(rows_window_list)



    def getWindowList(self):
        return self.__window_list
    

    def getWindowCount(self):
        return (self.__window_y_count, self.__window_x_count)
    
    def getInputImgPad(self):
        return self.__input_img_pad
    
    def getPadSize(self):
        pad_top = self.__pad_rows//2
        pad_bottom = self.__pad_rows//2 + (self.__pad_rows%2)
        pad_left = self.__pad_cols//2
        pad_right = self.__pad_cols//2 + (self.__pad_cols%2)
        return (pad_top, pad_bottom, pad_left, pad_right)