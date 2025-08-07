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
from utils.Normalize_Module import normalize
from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D
from SecteringMask import SecteringMask
from TrackFingerPrint import TrackFingerPrint



input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\056L9U.bmp"
input_files_path = glob(r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename/" + "*")

output_segment_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\segment\auto_selected_threshold_features_TV_0_5/"

for idx in range(len(input_files_path)):

    ########################## Read Input Image ###################
    input_file_name = input_files_path[idx]
    print(f"file name : {input_file_name}")
    input_img = cv.imread(input_files_path[idx])
    gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    weight_tv = 5.0
    mask_sector = SecteringMask(gray_img, weight_tv = weight_tv, preprocessing=False)
    mask_sector.MaskSector()

    filtered_img = mask_sector.getFilteredImg()
    sectors_list = mask_sector.getSectorsList()

    Tracking = TrackFingerPrint(filtered_img, sectors_list)
    Tracking.forward()
    output_img = Tracking.get_output_img()
    plt.imshow(output_img, cmap="gray")
    plt.show()
    

    base_filename = os.path.basename(input_file_name)
    output_file_name = output_segment_path + base_filename
    # cv.imwrite(output_file_name, output_img)
    # plt.imsave(output_file_name, output_img)
    # io.imsave(output_file_name, output_img)