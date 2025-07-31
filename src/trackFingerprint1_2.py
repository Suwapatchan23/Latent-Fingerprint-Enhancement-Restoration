import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io
from skimage.measure import regionprops, label
from skimage.exposure import equalize_adapthist
from scipy.ndimage import uniform_filter
from scipy.signal import windows
import skimage.morphology as skmorph
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from skimage.color import label2rgb
from scipy.signal import convolve2d, windows
from scipy.ndimage import binary_dilation

from utils.Normalize_Module import normalize, adjustRange
from utils.Windowing_Module import WindowPartition2D
from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D


########################### Function ######################
def dilate3D(mask, radius):
    """
    input:
        mask = mask binary image
        radius = radius from center of voxel
    return dilated image in 3D and structure
    """
    assert mask.ndim == 3
    mask = mask.astype(bool)
    padded = np.pad(mask, radius, mode='constant', constant_values=False)
    dilated = np.zeros_like(padded, dtype=bool)

    # loop z --> y --> x from -radius to radius +1 if radius is 1 --> (-1, 2) (3x3)
    for dz in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                shifted = np.roll(padded, shift=(dz, dy, dx), axis=(0,1,2))
                dilated |= shifted

    
    slices = tuple(slice(radius, -radius) for _ in range(3))

    return dilated[slices]


def select_fingerprint_area(energy_img_list, th):
    th = round(th, 2)
    energy_img_arr = np.array(energy_img_list)
    # normalize energy 
    energy_img_arr_norm = (energy_img_arr - energy_img_arr.min()) / (energy_img_arr.max() - energy_img_arr.min())
    segment_list = []
    temp_segment_list = []
    max_energy = energy_img_arr_norm.max()
    # print(th)
    threshold = max_energy * th
    # print(threshold)
    for i in range(len(energy_img_list)):
        
        segment_energy = np.where(energy_img_arr_norm[i] > threshold, energy_img_arr_norm[i], 0).astype(float)
        temp_energy = np.where(energy_img_arr_norm[i] > threshold, 1, 0).astype(float)
        # plt.imshow(temp_energy, cmap="gray")
        # plt.show()
        # print(i)
        # display_grid_image(temp_energy, block_size=33)
        segment_list.append(segment_energy)
        temp_segment_list.append(temp_energy)
        # plt.imshow(temp_energy, cmap="gray")
        # plt.show()
    paired_energy_list = []
    paired_segment_list = []
    for j in range(len(segment_list)):
        paired_energy = segment_list[j] * segment_list[(j+1)%18]
        paired_energy_list.append(paired_energy)
        binary_volume = np.where(paired_energy == 0, 0, 1)
        paired_segment_list.append(binary_volume)
        # print(j)
        # plt.figure()
        # plt.imshow(segment_list[j], cmap="gray")
        # plt.figure()
        # plt.imshow(segment_list[(j+1)%18], cmap="gray")
        # plt.figure()
        # plt.imshow(binary_volume, cmap="gray")
        # plt.show()
    
    paired_energy_list = np.array(paired_energy_list)
    paired_segment_list = np.array(paired_segment_list)
    temp_segment_list = np.array(temp_segment_list)

    # transpose to (y, x, z) from (z, y, x)
    paired_energy_list = paired_energy_list.transpose(1, 2, 0)
    paired_segment_list = paired_segment_list.transpose(1, 2, 0)
    temp_segment_list = temp_segment_list.transpose(1, 2, 0)

    # print(paired_energy_list.shape)
    
    ################################################################################################
    radius = 1
    size = 3
    dilated_volume = dilate3D(paired_segment_list, radius=radius)
    # for j in range(18):
        # print(j)
        # display_grid_image(paired_segment_list[:,:,j], block_size=33)
        # display_grid_image(dilated_volume[:,:,j], block_size=33)
        # plt.imshow(dilated_volume[:,:,j], cmap="gray")
        # plt.show()
    and_vol = np.logical_and(dilated_volume, temp_segment_list)
    # for j in range(18):
        # print(j)
        # display_grid_image(and_vol[:,:,j], block_size=33)
        # plt.imshow(and_vol[:,:,j], cmap="gray")
        # plt.show()
    # print(dilated_volume.shape)
    
    structure = np.ones((size, size, size), dtype=bool)
    labeled, num_obj = ndimage.label(and_vol, structure=structure)

    filtered_mask = np.zeros_like(labeled, dtype=bool)
    sum_segment = np.zeros_like(filtered_mask[:,:,0], dtype=bool)
    plotter = pv.Plotter()
    z_ranges = []

    for label_id in range(1, num_obj + 1):

        component = (labeled == label_id)
        z_coords = np.where(component)[2]
        z_range = z_coords.max() - z_coords.min() + 1 if z_coords.size > 0 else 0
        z_ranges.append((label_id, z_range))
        # print(z_range)
        filtered_mask |= component

        # visualization
        # verts, faces, _, _ = measure.marching_cubes(component, level=0.5, spacing=(1, 1, 10))
        # faces = np.hstack([[3, *f] for f in faces])
        # mesh = pv.PolyData(verts, faces)
        # plotter.add_mesh(mesh, color=np.random.rand(3), opacity=0.6)

    # plotter.show()
    n_sectors = 18
    for z in range(n_sectors):
        sum_segment += filtered_mask[:,:,z]

    return sum_segment
    # plt.show()
    ################################################################################################################

def gaussian_window_2d(shape, sigma):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return gauss

def display_grid_image(input_img, block_size):
    fig, ax = plt.subplots()
    ax.imshow(input_img, cmap='gray')
    grid_size = block_size
    ax.set_xticks(np.arange(-0.5, input_img.shape[1], grid_size))
    ax.set_yticks(np.arange(-0.5, input_img.shape[0], grid_size))
    ax.grid(color='red', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()

def find_energy(input_img, block_size):

    img = input_img.astype(np.float32)
    # avg each block
    mean = uniform_filter(img, size=block_size, mode='reflect')

    # avg squared each block
    mean_of_square = uniform_filter(img**2, size=block_size, mode='reflect')


    local_variance = mean_of_square - mean**2


    local_variance_display = cv.normalize(local_variance, None, 0, 255, cv.NORM_MINMAX)
    local_variance_display = local_variance_display.astype(np.uint8)


    return local_variance_display


def horizontal_mirror(segment):
    w = segment.shape[1]
    center = w // 2
    left = segment[:, :center]
    right = np.fliplr(left)

    return np.hstack([left, right])

def vertical_mirror(image):
    H = image.shape[0]
    center = H // 2
    top = image[:center, :]
    bottom = np.flipud(top)
    return np.vstack([top, bottom])

def mirror_segment(segment):
    h_mirror = horizontal_mirror(segment)
    full_mirror = vertical_mirror(h_mirror)
    return full_mirror

def select_largest_region(segment):
    labeled_mask = label(segment) 
    regions = regionprops(labeled_mask)

    # select largest area label
    if len(regions) > 0:
        largest_region = max(regions, key=lambda r: r.area)

        largest_mask = np.zeros_like(segment, dtype=bool)
        largest_mask[labeled_mask == largest_region.label] = 1
    else:
        largest_mask = np.zeros_like(segment)

    return largest_mask 

def compare_segments(segment1, segment2):

    segment1_area = np.count_nonzero(segment1)

    segment2_area = np.count_nonzero(segment2)

    diff = np.abs(segment2_area - segment1_area)

    return diff

def normalize_feature(values, eps=1e-8):
    min_val = np.min(values)
    max_val = np.max(values)
    return [(v - min_val) / (max_val - min_val + eps) for v in values]

def find_freq(img):
    plt.figure()
    plt.imshow(img, cmap="gray")
    FFT = Fourier2D(img)
    FFT.fft()
    magnitude = FFT.getMagnitude()
    FFT.showMagnitude()
    plt.show()
    peak_magnitude = magnitude.max()
    # find euclidean distance 
    y_pose, x_pose = np.where(magnitude == peak_magnitude)
    if len(y_pose) == 2:
        euclidean_distance = np.sqrt((x_pose[0] - x_pose[1])**2 + (y_pose[0] - y_pose[1])**2)
        freq = euclidean_distance / 2
    else:
        euclidean_distance = 0
        freq = 1
    return freq


def select_fingerprint_region(segment, input_img):
    labeled_mask = label(segment) 
    regions = regionprops(labeled_mask)
    fingerprint_mask = np.zeros_like(segment, dtype=bool)

    num_obj = len(regions)

    # filter only 3 largest components
    component_sizes = [(label_id, np.count_nonzero(labeled_mask == label_id)) for label_id in range(1, num_obj + 1)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    candidate_components = set([label for label,_ in component_sizes[:3]])

    scores = []
    variances_list = []
    edges_list = []
    compactness_list = []
    label_list = []
    area_list = []
    elongation_list = []

    for i in range(num_obj):
        curr_label = regions[i]
        if (curr_label.label) not in (candidate_components):
            continue
        if (curr_label.area) < 13000:
            continue
        # print(curr_label.area)
        curr_segment = np.zeros_like(segment, dtype=bool)
        curr_segment[labeled_mask == curr_label.label] = 1
        mask_img = input_img[curr_segment]
        area = curr_label.area
        perimeter = curr_label.perimeter

        # masked_img = np.zeros_like(input_img)
        # temp_img = input_img * curr_segment
        # bg_mean = mask_img.mean()
        # temp_img[curr_segment == 0.0] = bg_mean

        # sigma = 1.5   
        # filter = gaussian_window_2d(masked_img.shape, sigma=sigma)
        # temp_img = temp_img * filter
        # plt.imshow(curr_segment, cmap="gray")
        # plt.show()
        # print(np.var(mask_img))
        # freq = find_freq(temp_img)
        # print(freq)
        variance = np.var(mask_img)
        variance_ratio = variance / (curr_label.area)
        compactness = (perimeter ** 2) / area

        minr, minc, maxr, maxc = curr_label.bbox
        height = maxr - minr
        width = maxc - minc
        bbox_sizes = [height, width]
        elongation_ratio = max(bbox_sizes) / min(bbox_sizes)

        variances_list.append(variance)

        # compactness_list.append(compactness)
        # area_list.append(area)
        elongation_list.append(elongation_ratio)
        label_list.append(curr_label)
        # print(variance)
        # print(f"variance_ratio = {variance_ratio}")
        # print(f"compactness = {compactness}")
        # print(f"area = {area}")

    norm_variance = normalize_feature(variances_list)
    # print(f"norm_variance = {norm_variance}")
    # norm_compact = normalize_feature(compactness_list)
    # print(f"norm_compact = {norm_compact}")
    norm_elongation = normalize_feature(elongation_list)
    # print(f"norm_elongation = {norm_elongation}")
    # norm_area = normalize_feature(area_list)
    # print(f"norm_area = {norm_area}")

    # inverse score for some feature
    # norm_compact = [1 - c for c in norm_compact]
    norm_elongation = [1 - c for c in norm_elongation]
    
    scores = []
    for i in range(len(variances_list)):

        score = norm_variance[i] + norm_elongation[i]
        scores.append((score, label_list[i]))

    scores.sort(key= lambda x: x[0], reverse=True)
    top_score, fingerprint_label = scores[0]

    # print(scores)

    fingerprint_mask[labeled_mask == fingerprint_label.label] = 1
        

    return fingerprint_mask

###########################  Path #########################
# input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\049L3U.bmp"
raw_files_path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg/" + "*")
input_files_path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\sectoring_18_sectors/" + "*")
output_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\pixel_based_segment/"
output_path_2 = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\watershed_idea_segment/"
output_path_3 = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\spatial_energy\watershed_segment_fixed/"
temp_input = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg\080L8U.bmp.bmp"
output_segment = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\segment\auto_selected_threshold_features/"

raw_files_path.sort()
input_files_path.sort()

for idx in range(len(raw_files_path)):

    print(raw_files_path[idx])
    raw_img = cv.imread(raw_files_path[idx])
    raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    # raw_img = cv.imread(r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\002L3U.bmp")
    # raw_img = cv.imread(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg\003L8U.bmp")
    # raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    path = glob(input_files_path[idx] + "/" + "*")
    path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\sectoring_18_sectors\003L8U" + "/" + "*")
    # path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    energy_img_list = []
    # each sectors
    for i in range(len(path)):
        # print(path[i])
        input_img = cv.imread(path[i])
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

        block_size = 33

        energy = find_energy(gray_img, block_size=block_size)

        energy_img_list.append(energy)

    segments = []
    for th in np.arange(0.7, 0.0, -0.05):
        segment = select_fingerprint_area(energy_img_list, th=th)
        segments.append(segment)
        # labeled, num_labeled = ndimage.label(segment)
        # # plt.imshow(labeled)
        # # plt.show()
        # overlay = label2rgb(labeled, image=raw_img, bg_label=0, alpha=0.4)
        # plt.imshow(overlay)
        # # plt.show()
        # plt.pause(0.05)
    
    differences = []
    n = len(segments)
    for i in range(n-1):
        seg1 = segments[i]
        seg2 = segments[(i + 1)]  
        diff = compare_segments(seg1, seg2)  
        differences.append(diff)
        # first index (0) is the difference between index 0 and index 1

    max_diff = max(differences)
    idx_max_diff = differences.index(max_diff)
    print(f"threshold = {round(0.7 - idx_max_diff * 0.05, 2)}")
    # print(max_diff, idx_max_diff)
    # plt.figure()
    # plt.imshow(segments[idx_max_diff], cmap="gray")
    # plt.figure()
    # plt.imshow(segments[idx_max_diff+1], cmap="gray")

    selected_segment = segments[idx_max_diff]
        
    fillhole_segment = fillHoles(selected_segment.astype(np.float32))
    # plt.figure()
    # plt.imshow(fillhole_segment, cmap="gray")
    # largest_segment = select_largest_region(fillhole_segment)
    fingerprint_segment = select_fingerprint_region(fillhole_segment, raw_gray_img)

    output_img = raw_gray_img * fingerprint_segment
    # clahe_img = equalize_adapthist(output_img, (32, 32), clip_limit=0.08)
    # clahe_img = adjustRange(clahe_img, (0, 1), (0, 255)).astype(np.uint8)     # adjust range
    output_img[fingerprint_segment == 0] = raw_gray_img.mean()

    plt.figure()
    plt.imshow(output_img, cmap="gray")
    plt.show()


    base_filename = os.path.basename(raw_files_path[idx])
    output_file_name = output_segment + base_filename
    # cv.imwrite(output_file_name, output_img)
    # plt.imsave(output_file_name, output_img)
    # io.imsave(output_file_name, output_img)