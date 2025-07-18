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

from utils.Normalize_Module import normalize, adjustRange
from utils.Windowing_Module import WindowPartition2D
from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D


########################### Function ######################

def select_fingerprint_area(energy_img_list, th):
    th = round(th, 2)
    energy_img_arr = np.array(energy_img_list)
    segment_list = []
    temp_segment_list = []
    max_energy = energy_img_arr.max()
    print(th)
    threshold = max_energy * th
    # print(threshold)
    for i in range(len(energy_img_list)):
        segment_energy = np.where(energy_img_list[i] > threshold, energy_img_list[i], 0).astype(float)
        temp_energy = np.where(energy_img_list[i] > threshold, 1, 0).astype(float)
        # plt.imshow(temp_energy, cmap="gray")
        # plt.show()
        segment_list.append(segment_energy)
        temp_segment_list.append(temp_energy)
        # plt.imshow(temp_energy, cmap="gray")
        # plt.show()
    paired_energy_list = []
    paired_segment_list = []
    for j in range(len(segment_list)):
        paired_energy = segment_list[j] * segment_list[(j+1)%16]
        paired_energy_list.append(paired_energy)
        binary_volume = np.where(paired_energy == 0, 0, 1)
        paired_segment_list.append(binary_volume)
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

    structure = ndimage.generate_binary_structure(3, 2)  # 26-connected
    # structure = np.ones((3, 3, 3), dtype=bool)
    n_iterations = 5
    dilated_volume = ndimage.binary_dilation(paired_segment_list, structure=structure, iterations=n_iterations)
    # for j in range(18):
    #     # plt.imshow(dilated_volume[:,:,j], cmap="gray")
    #     # plt.show()
    and_vol = np.logical_and(dilated_volume, temp_segment_list)
    # for j in range(18):
        # plt.imshow(and_vol[:,:,j], cmap="gray")
        # plt.show()


    # print(dilated_volume.shape)
    labeled, num_obj = ndimage.label(and_vol, structure=structure)
    # filter only 3 largest components
    component_sizes = [(label_id, np.count_nonzero(labeled == label_id)) for label_id in range(1, num_obj + 1)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    candidate_components = set([label for label,_ in component_sizes[:3]])

    filtered_mask = np.zeros_like(labeled, dtype=bool)
    sum_segment = np.zeros_like(filtered_mask[:,:,0], dtype=bool)
    plotter = pv.Plotter()
    z_ranges = []

    for label_id in range(1, num_obj + 1):
        # 1 st condition
        # if (label_id) not in (candidate_components):
        #     continue
        component = (labeled == label_id)

        z_coords = np.where(component)[2]
        z_range = z_coords.max() - z_coords.min() + 1 if z_coords.size > 0 else 0

        # if (z_range < 4):
        #     continue

        z_ranges.append((label_id, z_range))
        # print(z_range)
        filtered_mask |= component

        # print(filtered_mask.shape)
        # mask = (labeled == label_id)
        # z_coords = np.where(mask)[2]  
        # z_range = z_coords.max() - z_coords.min() + 1 if z_coords.size > 0 else 0
        # print(f"Component {label_id}: z-range = {z_range} slices")

        # visualization
        # verts, faces, _, _ = measure.marching_cubes(component, level=0.5, spacing=(1, 1, 10))
        # faces = np.hstack([[3, *f] for f in faces])
        # mesh = pv.PolyData(verts, faces)
        # plotter.add_mesh(mesh, color=np.random.rand(3), opacity=0.6)


    ############### use compactness and elongation to filter ###################
    labeled_2, _ = ndimage.label(filtered_mask, structure=structure)
    # print("Number of labels:", np.max(labeled_2))
    regions = regionprops(labeled_2.astype(np.uint8))
    best_compact = None
    best_score = 0

    # 2 nd condition
    best_elongation = None
    best_elongation_score = float("inf")
    # print(len(regions))
    for region in regions:
        volume = region.area
        minr, minc, mind, maxr, maxc, maxd = region.bbox
        depth = maxd - mind
        height = maxr - minr
        width = maxc - minc
        bbox_volume = (maxr - minr) * (maxc - minc) * (maxd - mind)
        bbox_sizes = [depth, height, width]
        elongation_ratio = max(bbox_sizes) / min(bbox_sizes)
        if volume == 0:
            continue
        compact_ratio = volume / bbox_volume 
        # print(compact_ratio)
        curr_label = region.label
        # compact_list.append([curr_label, compact_ratio])
        if compact_ratio > best_score:
            best_score = compact_ratio
            best_compact = curr_label

        # print(elongation_ratio)
        if elongation_ratio < best_elongation_score:
            best_elongation_score = elongation_ratio
            best_elongation = curr_label

    # sort from higher to lower
    # compact_list.sort(key = lambda x: x[1])
    # # print(compact_list)
    # compact_list.pop(0)
    # # print(compact_list)

    # labels, compact_nums = zip(*compact_list)
    # # print(len(labels))

    # filltered_components =  np.isin(labeled_2, labels)

    most_compact_component = (labeled_2 == best_compact)
    least_elongation_component = (labeled_2 == best_elongation)
    n_sectors = 18
    for z in range(n_sectors):
        sum_segment += filtered_mask[:,:,z]
    # for z in range(n_sectors):
    #     sum_segment += most_compact_component[:,:,z]
    # for z in range(n_sectors):
    #     sum_segment += least_elongation_component[:,:,z]
    # plotter.show()
    return sum_segment
    # plt.show()
    ################################################################################################################


def display_grid_image(input_img, block_size):

    h,w = input_img.shape

    img_grid = input_img.copy()

    for y in range(0, h, block_size):
        cv.line(img_grid, (0, y), (w, y), (0, 255, 0), 1)

    for x in range(0, w, block_size):
        cv.line(img_grid, (x, 0), (x, h), (0, 255, 0), 1)

    plt.figure()
    plt.imshow(img_grid, cmap="gray")
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


###########################  Path #########################
# input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\049L3U.bmp"
raw_files_path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg/" + "*")
input_files_path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\sectoring_18_sectors/" + "*")
output_path = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\pixel_based_segment/"
output_path_2 = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\watershed_idea_segment/"
output_path_3 = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\spatial_energy\watershed_segment_fixed/"
temp_input = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\fillteredImg\080L8U.bmp.bmp"
output_segment = r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\segment\org_th_0_3/"

raw_files_path.sort()
input_files_path.sort()

for idx in range(len(raw_files_path)):

    print(raw_files_path[idx])
    # raw_img = cv.imread(raw_files_path[idx])
    # raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    raw_img = cv.imread(r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\002L3U.bmp")
    raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    # path = glob(input_files_path[idx] + "/" + "*")
    # path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    path = glob(r"D:\KSIP_Research\Latent\Latent_Fingerprint_Enhancement_Restoration\output\sectoring_18_sectors\002L3U" + "/" + "*")

    energy_img_list = []
    # each sectors
    for i in range(len(path)):

        input_img = cv.imread(path[i])
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

        block_size = 33

        energy = find_energy(gray_img, block_size=block_size)

        energy_img_list.append(energy)

    for th in np.arange(0.9, 0.0, -0.05):
        segment = select_fingerprint_area(energy_img_list, th=th)
        labeled, num_labeled = ndimage.label(segment)
        # plt.imshow(labeled)
        # plt.show()
        overlay = label2rgb(labeled, image=raw_img, bg_label=0, alpha=0.4)
        plt.imshow(overlay)
        # plt.show()
        plt.pause(0.05)


    fillhole_segment = fillHoles(segment.astype(np.float32))
    largest_segment = select_largest_region(fillhole_segment)

    output_img = raw_gray_img * largest_segment
    # clahe_img = equalize_adapthist(output_img, (32, 32), clip_limit=0.08)
    # clahe_img = adjustRange(clahe_img, (0, 1), (0, 255)).astype(np.uint8)     # adjust range
    output_img[largest_segment == 0] = raw_gray_img.mean()

    # plt.figure()
    # plt.imshow(output_img, cmap="gray")
    # plt.show()


    base_filename = os.path.basename(raw_files_path[idx])
    output_file_name = output_segment + base_filename
    # cv.imwrite(output_file_name, output_img)
    # plt.imsave(output_file_name, output_img)
    # io.imsave(output_file_name, output_img)