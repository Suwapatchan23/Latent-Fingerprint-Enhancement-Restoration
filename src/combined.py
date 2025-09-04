import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from scipy.signal import convolve2d, windows
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
import pyvista as pv
from scipy.ndimage import uniform_filter
from skimage.measure import regionprops, label




from utils.Filter_Module import bandpassFilter
from utils.Fourier_Module import Fourier2D
from utils.Normalize_Module import normalize, adjustRange
from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D
from SecteringMask import SecteringMask
from TrackFingerPrint import TrackFingerPrint
from utils.Fourier_Module import STFT
from utils.General_Module import findMaxPeak, findQuality




input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\082L6U.bmp"
input_files_path = glob(r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename/" + "*")

output_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\Thesis\segment_and_enh_spectralgaborfilterbank_Adaptive_Bilateral_Gradient/"

for idx in range(len(input_files_path)):

    ########################## Read Input Image ###################
    input_file_name = input_files_path[idx]
    # input_file_name = input_file_path
    base_filename = os.path.basename(input_file_name)
    output_file_name = output_path + base_filename
    # if os.path.exists(output_file_name):
    #     continue
    print(f"file name : {input_file_name}")

    input_img = cv.imread(input_file_name)
    gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    weight_tv = 10.0
    mask_sector = SecteringMask(gray_img, weight_tv = weight_tv, preprocessing=True)
    mask_sector.MaskSector()

    filtered_img = mask_sector.getFilteredImg()
    sectors_list = mask_sector.getSectorsList()

    Tracking = TrackFingerPrint(filtered_img, sectors_list)
    Tracking.forward()
    output_img = Tracking.get_output_img()
    output_segment = Tracking.get_output_segment()

    ################### windowing and thresholding #####################
    overlap_size = 32
    non_overlap_size = 32
    stft = STFT(output_img, overlap_size=overlap_size, non_overlap_size=non_overlap_size)
    stft.forward()
    stft_magnitude_list = stft.getMagnitudeList()
    # stft.showSpatial()
    # stft.showMagnitude()
    # plt.show()

    max_peak, _ = findMaxPeak(stft_magnitude_list)
    FRQ_list, global_max_peak = findQuality(max_peak)
    FRQ_img = stft.getQuality(FRQ_list)
    
    t_FRQ = 0.25

    threshold_FRQ = np.where(FRQ_list > t_FRQ, FRQ_list, 0)
    binary_threshold_FRQ = np.where(FRQ_list > t_FRQ, 1, 0)
    threshold_FRQ_img = stft.getQuality(threshold_FRQ)
    binary_threshold_FRQ_img = stft.getQuality(binary_threshold_FRQ)
    binary_threshold_FRQ_img = adjustRange(binary_threshold_FRQ_img, (0,255), (0,1)).astype(np.uint8)

    fillhole_segment = fillHoles(binary_threshold_FRQ_img.astype(np.float32))
    fillhole_output = fillhole_segment * output_img
    # candidate = select top N regions
    largest_segment = Tracking.select_largest_region(fillhole_segment)
    candidate_segment = Tracking.enh_img(largest_segment)
    enh_img = Tracking.get_enh_img()

    # plt.figure()
    # plt.imshow(output_img, cmap='gray')       
    # plt.imshow(FRQ_img, cmap='hot', alpha=0.5)
    # plt.colorbar(label='Quality')
    # plt.figure()
    # plt.imshow(output_img, cmap='gray')       
    # plt.imshow(threshold_FRQ_img, cmap='hot', alpha=0.5)
    # plt.colorbar(label='Quality')
    # plt.figure()
    # plt.imshow(fillhole_output, cmap='gray')   
    # plt.figure()
    # plt.imshow(candidate_segment, cmap="gray")
    # plt.show()
    plt.figure()
    plt.imshow(enh_img, cmap="gray")
    plt.show()

    # base_filename = os.path.basename(input_file_name)
    # output_file_name = output_path + base_filename
    # cv.imwrite(output_file_name, enh_img)
    # plt.imsave(output_file_name, enh_img)
    # io.imsave(output_file_name, enh_img)