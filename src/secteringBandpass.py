import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from scipy.signal import convolve2d, windows


from utils.Filter_Module import bandpassFilter
from utils.Fourier_Module import Fourier2D
from utils.Normalize_Module import normalize

########################### Function ###########################
def get_sector_mask(shape, center, r_inner, r_outer, angle_mask, angle_mask_mirror):
    h, w = shape
    x = np.arange(w) - center[0]
    y = np.arange(h) - center[1]
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    Theta = (Theta + 2*np.pi) % (2*np.pi)
    
    mask = ((R >= r_inner) & (R <= r_outer) & angle_mask(Theta)) | ((R >= r_inner) & (R <= r_outer) & angle_mask_mirror(Theta))
    return mask

def mask_sector(magnitude_img, r_inner=37, r_outer=183, num_sectors=18, sector_index=0, overlap_deg = 0.0):

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
    
    mask = get_sector_mask(magnitude_img.shape, center, r_inner, r_outer, angle_mask, angle_mask_mirror)


    return mask

def applied_gaussian(mask, kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize the kernel

    filter = convolve2d(mask, kernel, mode="same")
    return filter

def save_img(output_img, output_dir_name):
    output_file_name = output_dir_name + ".bmp"
    cv.imwrite(output_file_name, output_img)
    plt.imsave(output_file_name, output_img)
    io.imsave(output_file_name, output_img)

def makeDirs(input_file, output_path):
    base_filename = os.path.basename(input_file).split('.')[0]
    dir_name = output_path + base_filename
    if not os.path.exists(dir_name):  # create folder if folder don't exists
        os.makedirs(dir_name)
    dir_name = dir_name + "/"
    return dir_name



###########################  Path #########################

input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\056L9U.bmp"
input_files_path = glob(r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename/" + "*")

output_dir_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\sectoring_18_sectors/"
output_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg_pre/"
for idx in range(len(input_files_path)):

    ########################## Read Input Image ###################
    input_file_name = input_files_path[idx]
    print(f"file name : {input_file_name}")
    input_img = cv.imread(input_files_path[idx])
    gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)



    ######################### Pre-processing ################################
    # output_img = gray_img.copy()

    # # คำนวณค่าเฉลี่ยของภาพ (หรือนอกกล่องก็ได้)
    # global_mean = int(np.mean(gray_img))

    # # กำหนด threshold ความดำเข้ม
    # threshold = 40

    # # วน loop ทั่วภาพ โดยไม่รวมขอบ (เพราะ 3x3)
    # for y in range(1, gray_img.shape[0] - 1):
    #     for x in range(1, gray_img.shape[1] - 1):
    #         patch = gray_img[y-1:y+2, x-1:x+2]

    #         # เช็คว่าใน 3x3 นี้ ทุกจุด "ดำเข้ม"
    #         if np.all(patch <= threshold):
    #             output_img[y-1:y+2, x-1:x+2] = global_mean  # แทนด้วยค่าเฉลี่ย
    # kernel_size = 3
    # sigma = 1.5
    # output_img = applied_gaussian(output_img, kernel_size=kernel_size, sigma=sigma)
    ######################## Tukey Window ###################################
    h, w = gray_img.shape
    # print(h, w)
    alpha = 0.2
    tukey_y = windows.tukey(h, alpha).reshape(-1, 1)
    tukey_x = windows.tukey(w, alpha).reshape(1, -1)
    tukey_2d = tukey_y @ tukey_x

    tukey_img = gray_img * tukey_2d

    # plt.figure()
    # plt.imshow(gray_img, cmap="gray")
    # plt.figure()
    # plt.imshow(tukey_img, cmap="gray")
    # plt.show()

    ########################## Fourier Transform ############################
    fourier = Fourier2D(tukey_img)
    fourier.fft()

    ########################## Bandpass Filter ##############################
    filter_size = input_img.shape
    filter_pos = (filter_size[0] // 2, filter_size[1] //2)
    # filter_center = 110
    # filter_bw = 145
    filter_center = 108
    filter_bw = 96
    bp_filter = bandpassFilter(filter_size, filter_pos, filter_center, filter_bw, "Gaussian")
    magnitude = fourier.getMagnitude()
    # plt.figure()
    # plt.imshow(magnitude, cmap="hot")
    # plt.figure()
    # plt.imshow(bp_filter, cmap="hot")
    # plt.show()
    filtered_magnitude = bp_filter * magnitude
    plt.figure()
    plt.imshow(filtered_magnitude, cmap="hot")
    
    fourier.setMagnitude(filtered_magnitude)
    fourier.ifft()

    filtered_img = fourier.getOutputImage()
    plt.figure()
    plt.imshow(filtered_img, cmap="gray")
    plt.show()
    filtered_img = normalize(filtered_img)

    base_filename = os.path.basename(input_file_name).split('.')[0]
    dir_name = output_path + base_filename
    # save_img(filtered_img, dir_name)

    ######################### Sectoring ###########################
    combined_mask = np.zeros_like(filtered_magnitude, dtype=np.float64)
    # fill numbers in index_list manually to combine sectors
    index_list = [0,1,2,3,4,5,6,9,10,11,12,13,14]

    n_sectors = 18
    overlap_deg = 0
    need_combine_mask = False

    kernel_size = 11
    sigma = 1.5

    output_dir_name = makeDirs(input_file_name, output_dir_path)
    # temp = filtered_magnitude.copy()
    for i in range (n_sectors):
        masked_magnitude = filtered_magnitude.copy()
        mask = mask_sector(filtered_magnitude, sector_index=i, overlap_deg=overlap_deg)
        mask = applied_gaussian(mask, kernel_size=kernel_size, sigma=sigma)

        # print(i)
        if need_combine_mask:
            if (i in index_list):
                combined_mask += mask
                # plt.figure()
                # plt.imshow(combined_magnitude, cmap="hot")
        # masked_freq = np.zeros_like(filtered_magnitude)
        # masked_freq[mask] = filtered_magnitude[mask]
        masked_magnitude = mask * filtered_magnitude

        # temp += masked_magnitude

        # plt.figure()
        # plt.imshow(filtered_magnitude, cmap="hot")
        # plt.figure()
        # plt.imshow(masked_magnitude, cmap="hot")
      
        fourier.setMagnitude(masked_magnitude)
        fourier.ifft()
        output_img = fourier.getOutputImage()
        norm_img = normalize(output_img)
        # plt.figure()
        # plt.imshow(norm_img, cmap="gray")
        

        output_dir = output_dir_name + str(i)
        # save_img(norm_img, output_dir)

        # plt.figure()
        # plt.imshow(mask, cmap="gray")
        # plt.figure()
        # plt.imshow(masked_magnitude, cmap="hot")
        # # plt.figure()
        # plt.figure()
        # plt.imshow(norm_img, cmap="gray")

        # plt.show()
    # plt.imshow(temp, cmap="hot")
    # plt.show()


    if need_combine_mask:
        combined_magnitude = filtered_magnitude.copy()
        combined_magnitude = combined_mask * filtered_magnitude

        fourier.setMagnitude(combined_magnitude)
        fourier.ifft()
        output_img = fourier.getOutputImage()
        output_img = normalize(output_img)
        plt.figure()
        plt.imshow(filtered_img, cmap="gray")
        plt.figure()
        plt.imshow(combined_mask, cmap="gray")
        plt.figure()
        plt.imshow(combined_magnitude, cmap="hot")
        plt.figure()
        plt.imshow(output_img, cmap="gray")
        plt.show()
