import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from scipy.signal import convolve2d, windows
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import uniform_filter
import pyvista as pv
from scipy import ndimage



from utils.Filter_Module import bandpassFilter
from utils.Fourier_Module import Fourier2D
from utils.Normalize_Module import normalize
from utils.Windowing_Module import WindowPartition2D


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
    
    def display_grid_image(self, input_img, block_size):
        fig, ax = plt.subplots()
        ax.imshow(input_img, cmap='gray')
        grid_size = block_size
        ax.set_xticks(np.arange(-0.5, input_img.shape[1], grid_size))
        ax.set_yticks(np.arange(-0.5, input_img.shape[0], grid_size))
        ax.grid(color='red', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    def _find_sigma(self, gray_img):
        max_gradient_list = []
        window = WindowPartition2D(gray_img, 32, 32)
        window.forward()
        window_list = window.getWindowList()
        window_y, window_x = window.getWindowCount()
        grad_mag_list = np.zeros_like(window_list)
        for y in range(window_y):
            for x in range(window_x):
                gx, gy = np.gradient(window_list[y][x])
                grad_mag = np.sqrt(gx**2 + gy**2)
                # plt.figure()
                # plt.imshow(window_list[y][x], cmap="gray")
                # plt.figure()
                # plt.imshow(grad_mag, cmap="hot")
                # plt.show()
                local_max_gradient = np.max(grad_mag)
        
                grad_mag_list[y][x] = grad_mag
                max_gradient_list.append(local_max_gradient)

        max_gradient_list = np.reshape(max_gradient_list, (window_y,window_x))
        global_max_gradint = np.max(max_gradient_list)
        # print(global_max_gradint)

        # self.display_grid_image(gray_img, 32)
        # plt.figure()
        # # plt.imshow(gray_img, cmap="gray")
        # plt.imshow(max_gradient_list, cmap="hot")
        # plt.colorbar(label='Quality')

        idx_block_y, idx_block_x = np.where(max_gradient_list == global_max_gradint)
        idx_block_y = idx_block_y[0]
        idx_block_x = idx_block_x[0]
        block = window_list[idx_block_y][idx_block_x]
        # plt.figure()
        # plt.imshow(block, cmap="gray")
        _, th = cv.threshold(block, 40, 255, cv.THRESH_TOZERO)
        # plt.figure()
        # plt.imshow(th, cmap="gray")
        # plt.show()


        # max_intensity = np.max(th[th != 0])
        max_pos = np.unravel_index(np.argmax(grad_mag_list[idx_block_y,idx_block_x]), grad_mag_list[idx_block_y,idx_block_x].shape)
        min_pos = np.unravel_index(np.argmin(grad_mag_list[idx_block_y,idx_block_x]), grad_mag_list[idx_block_y,idx_block_x].shape)
        max_grad_intensity = block[max_pos]
        min_grad_intensity = block[min_pos]
        max_intensity = np.max(th)
        min_intensity = np.min(th[th != 0])

        # print(max_intensity, min_intensity)

        # print(max_grad_intensity, min_intensity)

        # max_pos = np.unravel_index(np.argmax(grad_mag_list[idx_block_y,idx_block_x]), grad_mag_list[idx_block_y,idx_block_x].shape)
        # min_gradient = np.min(grad_mag_list[idx_block_y,idx_block_x])
        # min_pos = np.unravel_index(np.argmin(grad_mag_list[idx_block_y,idx_block_x]), grad_mag_list[idx_block_y,idx_block_x].shape)
        # diff_intensity = np.abs((np.max(window_list[idx_block_y][idx_block_x])).astype(np.float32) - (np.min(window_list[idx_block_y][idx_block_x])).astype(np.float32))

        diff_intensity = np.abs(max_intensity.astype(np.float32) - min_intensity.astype(np.float32))
        d = np.abs(max_grad_intensity.astype(np.float32) - min_grad_intensity.astype(np.float32))

        # m = 0
        # if diff_intensity < 200:
        #     print("different intensity less than 200")
        #     sigma_intensity = 12
        #     return sigma_intensity
        # # elif diff_intensity > 120:
        # #     m = 4
        # else:
        #     m = 5
        m = 5
        # sigma_intensity = int(diff_intensity / m)
        sigma_intensity = int(diff_intensity / m)
        if sigma_intensity == 0:
            sigma_intensity = 10
        # sigma_intensity = 10
        # print(f"different intensity = {d}")
        # print(f"sigma_intensity = {sigma_intensity}")
        return sigma_intensity


    def _TV_preprocessing(self, img):
        input_img_float = img.astype(np.float64)
        cartoon_img = denoise_tv_chambolle(input_img_float, weight=self.weight_tv)
        texture_img = input_img_float - cartoon_img.astype(np.float64)
        texture_img = normalize(texture_img).astype(np.uint8)
        
        return texture_img
    
    def _bilateral_preprocessing(self, img, sigma_intensity, sigma_space=8):
        input_img_float = img.astype(np.float32)
        # 40 10
        cartoon_img  = cv.bilateralFilter(input_img_float, d=0, sigmaColor=sigma_intensity, sigmaSpace=sigma_space)
        # plt.figure()
        # plt.imshow(img, cmap="gray")
        # plt.figure()
        # plt.imshow(cartoon_img, cmap="gray")
        texture_img = input_img_float - cartoon_img.astype(np.float64)
        texture_img = normalize(texture_img).astype(np.uint8)
        plt.figure()
        plt.imshow(texture_img, cmap="gray")
        plt.show()
        return texture_img
    
    def _applied_tukey_window(self, img, alpha=0.2):
        h, w = img.shape
        tukey_y = windows.tukey(h, alpha).reshape(-1, 1)
        tukey_x = windows.tukey(w, alpha).reshape(1, -1)
        tukey_2d = tukey_y @ tukey_x

        tukey_img = img * tukey_2d

        return tukey_img
    
    def _estimate_noise(self, img):
        diff = np.abs(img - np.median(img))
        # to ignore over peak value
        diff = np.clip(diff, 0, 20)  
        sigma_noise = 1.4826 * np.median(diff)
        print(sigma_noise)
        return sigma_noise
    
    
    def MaskSector(self):

        if self.need_preprocessing :
            # tv_img = self._TV_preprocessing(self.gray_img)
            sigma_intensity = self._find_sigma(self.gray_img)
            # img_norm = self.gray_img.astype(float)/255.0
            # sigmaColor_norm = self._estimate_noise(img_norm)
            # sigmaColor = sigmaColor_norm * 255

            # print(sigmaColor)
            bilateral_img = self._bilateral_preprocessing(self.gray_img, sigma_intensity)
            pre_img = bilateral_img
            # var = np.var(bilateral_img)
            # print(var)
            # if var < 600:
            #     pre_img = tv_img
            # else:
            #     pre_img = bilateral_img
        else:
            pre_img = self.gray_img
        
        # plt.figure()
        # plt.imshow(self.gray_img, cmap="gray")
        # plt.figure()
        # plt.imshow(tv_img, cmap="gray")
        # plt.figure()
        # plt.imshow(bilateral_img, cmap="gray")
        # plt.show()
        tukey_img = self._applied_tukey_window(pre_img)

        fourier = Fourier2D(tukey_img)
        fourier.fft()

        filter_size = self.gray_img.shape
        filter_pos = (filter_size[0] // 2, filter_size[1] //2)
        #filter center = 108, filter bandwidth = 96
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

        # plt.figure()
        # plt.imshow(filtered_img, cmap="gray")
        # plt.show()

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






