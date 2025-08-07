import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from scipy.signal import convolve2d, windows
from skimage.restoration import denoise_tv_chambolle




from utils.Filter_Module import bandpassFilter
from utils.Fourier_Module import Fourier2D
from utils.Normalize_Module import normalize


class SecteringMask:


    def __init__(self, gray_img, weight_tv, preprocessing):
        self.gray_img = gray_img
        self.weight_tv = weight_tv
        self.filtered_img = None
        self.sectors_list = []
        self.need_preprocessing = preprocessing


    def _get_sector_mask(self, shape, center, r_inner, r_outer, angle_mask, angle_mask_mirror):
        h, w = shape
        x = np.arange(w) - center[0]
        y = np.arange(h) - center[1]
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        Theta = (Theta + 2*np.pi) % (2*np.pi)
        
        mask = ((R >= r_inner) & (R <= r_outer) & angle_mask(Theta)) | ((R >= r_inner) & (R <= r_outer) & angle_mask_mirror(Theta))
        return mask

    def _mask_sector(self, magnitude_img, r_inner=37, r_outer=183, num_sectors=18, sector_index=0, overlap_deg = 0.0):

        h, w = magnitude_img.shape
        center = (w // 2, h // 2)

        overlap_radian = (overlap_deg * 1 * np.pi) / 180
            
        sector_size = 1 * np.pi / num_sectors
        start_angle = ((sector_index * sector_size)) % (2 * np.pi)
        end_angle = (((sector_index + 1) * sector_size)) % (2 * np.pi)



        # in case start_angle > end_angle
        if start_angle < end_angle:
            angle_mask = lambda theta: (theta >= start_angle) & (theta < end_angle)
        else:
            angle_mask = lambda theta: (theta >= start_angle) | (theta < end_angle)

        # mirror_thata
        start_mirror_angle = (start_angle + np.pi) % (2 * np.pi)
        end_mirror_angle = (end_angle + np.pi) % (2 * np.pi)
        # print(start_angle, end_angle)
        # print(start_mirror_angle, end_mirror_angle)

        # in case start_angle > end_angle for mirror_theta
        if start_mirror_angle < end_mirror_angle:
            angle_mask_mirror = lambda theta: (theta >= start_mirror_angle) & (theta < end_mirror_angle)
        else:
            angle_mask_mirror = lambda theta: (theta >= start_mirror_angle) | (theta < end_mirror_angle)
        
        mask = self._get_sector_mask(magnitude_img.shape, center, r_inner, r_outer, angle_mask, angle_mask_mirror)


        return mask

    def _applied_gaussian(self, mask, kernel_size, sigma):
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize the kernel

        filter = convolve2d(mask, kernel, mode="same")
        return filter

    def save_img(self, output_img, output_dir_name):
        output_file_name = output_dir_name + ".bmp"
        cv.imwrite(output_file_name, output_img)
        plt.imsave(output_file_name, output_img)
        io.imsave(output_file_name, output_img)

    def makeDirs(self, input_file, output_path):
        base_filename = os.path.basename(input_file).split('.')[0]
        dir_name = output_path + base_filename
        if not os.path.exists(dir_name):  # create folder if folder don't exists
            os.makedirs(dir_name)
        dir_name = dir_name + "/"
        return dir_name
    
    def _pre_processing(self, img):
        input_img_float = img.astype(np.float64)
        cartoon_img = denoise_tv_chambolle(input_img_float, weight=self.weight_tv)
        texture_img = input_img_float - cartoon_img.astype(np.float64)
        texture_img = normalize(texture_img).astype(np.uint8)
        return texture_img

    def _applied_tukey_window(self, img, alpha=0.2):
        h, w = img.shape
        tukey_y = windows.tukey(h, alpha).reshape(-1, 1)
        tukey_x = windows.tukey(w, alpha).reshape(1, -1)
        tukey_2d = tukey_y @ tukey_x

        tukey_img = img * tukey_2d

        return tukey_img
    
    
    def MaskSector(self):
    
        if self.need_preprocessing :
            pre_img = self._pre_processing(self.gray_img)
        else:
            pre_img = self.gray_img
        tukey_img = self._applied_tukey_window(pre_img)

        fourier = Fourier2D(tukey_img)
        fourier.fft()

        filter_size = self.gray_img.shape
        filter_pos = (filter_size[0] // 2, filter_size[1] //2)
        filter_center = 108
        filter_bw = 96
        bp_filter = bandpassFilter(filter_size, filter_pos, filter_center, filter_bw, "Gaussian")
        magnitude = fourier.getMagnitude()
        filtered_magnitude = bp_filter * magnitude

        
        fourier.setMagnitude(filtered_magnitude)
        fourier.ifft()

        filtered_img = fourier.getOutputImage()

        filtered_img = normalize(filtered_img)
        self.filtered_img = filtered_img

        ######################### Sectoring ###########################
        combined_mask = np.zeros_like(filtered_magnitude, dtype=np.float64)
        # fill numbers in index_list manually to combine sectors
        index_list = [0,1,2,3,4,5,6,9,10,11,12,13,14]

        n_sectors = 18
        overlap_deg = 0
        need_combine_mask = False

        kernel_size = 11
        sigma = 1.5

        # temp = filtered_magnitude.copy()
        for i in range (n_sectors):
            masked_magnitude = filtered_magnitude.copy()
            mask = self._mask_sector(filtered_magnitude, sector_index=i, overlap_deg=overlap_deg)
            mask = self._applied_gaussian(mask, kernel_size=kernel_size, sigma=sigma)


            if need_combine_mask:
                if (i in index_list):
                    combined_mask += mask

            masked_magnitude = mask * filtered_magnitude
        
            fourier.setMagnitude(masked_magnitude)
            fourier.ifft()
            output_img = fourier.getOutputImage()
            norm_img = normalize(output_img)

            self.sectors_list.append(norm_img)

    def getSectorsList(self):
        return self.sectors_list
    
    def getFilteredImg(self):
        return self.filtered_img






