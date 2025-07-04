import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from skimage.measure import regionprops, label
from skimage.exposure import equalize_adapthist

from utils.Normalize_Module import normalize, adjustRange
from utils.Windowing_Module import WindowPartition2D


########################### Function ######################

def select_fingerprint_area(energy_img_list):

    combined_segment_img = np.zeros_like(energy_img_list[0], dtype=bool)

    for i in range(len(energy_img_list)):
        combined_img = energy_img_list[i] + energy_img_list[(i+1)%8]
        # plt.figure()
        # plt.imshow(combined_img, cmap="hot")

        max_energy = combined_img.max()
        threshold = max_energy * 0.40
        seg_max_energy_img = np.where(combined_img > threshold, 1, 0).astype(np.uint8)
        # plt.figure()
        # plt.imshow(seg_max_energy_img, cmap="gray")

        labeled_mask = label(seg_max_energy_img) 
        regions = regionprops(labeled_mask)

        # select largest area label
        if len(regions) > 0:
            largest_region = max(regions, key=lambda r: r.area)
            largest_mask = np.zeros_like(seg_max_energy_img, dtype=bool)
            largest_mask[labeled_mask == largest_region.label] = 1
        else:
            largest_mask = np.zeros_like(seg_max_energy_img) 
        
        # plt.figure()
        # plt.imshow(largest_mask, cmap="gray")

        # plt.show()

        combined_segment_img |= largest_mask

        # plt.figure()
        # plt.imshow(combined_segment_img, cmap="gray")

        # plt.show()
        

    return combined_segment_img


###########################  Path #########################
# input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\049L3U.bmp"
raw_files_path = glob(r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\fillteredImg/" + "*")
input_files_path = glob(r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\sectoring_8_sectors/" + "*")
output_path = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\extract_from_energy_map/"

raw_files_path.sort()
input_files_path.sort()

for idx in range(len(raw_files_path)):

    raw_img = cv.imread(raw_files_path[idx])
    raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    path = glob(input_files_path[idx] + "/" + "*")
    path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # print(raw_files_path[idx])

    energy_img_list = []
    for i in range(len(path)):

        # print(path[i])
        input_img = cv.imread(path[i])
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

        ################################ Windowing #########################
        overlapped_size = 64
        non_overlapped_size = 32


        window = WindowPartition2D(gray_img, overlapped_size, non_overlapped_size)
        window.forward()
        window_list = window.getWindowList()
        window_y_count , window_x_count = window.getWindowCount()
        energy_map = []

        for y in range(window_y_count):
            for x in range(window_x_count):

                block = window_list[y][x]
                # plt.imshow(block, cmap="gray")
                # plt.show()
                variance = np.var(block)
                energy_map.append(variance)

        h, w = gray_img.shape

        energy_map = np.reshape(energy_map, (window_y_count,window_x_count))
        energy_img = cv.resize(energy_map, (w,h), interpolation=cv.INTER_AREA)

        # plt.imshow(energy_img, cmap="hot")
        # plt.show()

        # collect energy image for each sectors
        energy_img_list.append(energy_img)

    segment = select_fingerprint_area(energy_img_list)

    output_img = segment * raw_gray_img

    # CLAHE
    # enh_img = equalize_adapthist(output_img, (32, 32), clip_limit=0.08)
    # enh_img = adjustRange(enh_img, (0, 1), (0, 255))     # adjust range

    plt.figure()
    plt.imshow(raw_gray_img, cmap="gray")
    plt.figure()
    plt.imshow(output_img, cmap="gray")
    plt.show()


    ################## save files #########################
    base_filename = os.path.basename(raw_files_path[idx])
    output_file_name = output_path + base_filename
    # cv.imwrite(output_file_name, output_img)
    # plt.imsave(output_file_name, output_img)
    # io.imsave(output_file_name, output_img)