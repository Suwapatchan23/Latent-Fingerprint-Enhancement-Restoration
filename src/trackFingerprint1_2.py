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

from utils.Normalize_Module import normalize, adjustRange
from utils.Windowing_Module import WindowPartition2D
from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D


########################### Function ######################

def gradient(energy_img_list):
    combined_segment_img = np.zeros_like(energy_img_list[0], dtype=bool)

    # combined_img = np.zeros_like(energy_img_list[0])
    for i in range(len(energy_img_list)):
        energy_stack = np.stack([energy_img_list[(i-2)%16], energy_img_list[(i-1)%16], energy_img_list[i], energy_img_list[(i+1)%16], energy_img_list[(i+2)%16]], axis=-1)  # shape = (H, W, 3)
        # combined_img = energy_img_list[(i-1)%8].astype(float) + energy_img_list[i].astype(float) + energy_img_list[(i+1)%8].astype(float)
        # combined_img = normalize(combined_img)
        # output_energy = gaussian_filter(energy_stack.astype(float), sigma=1)

        # box filter to smooth first
        size_block = (7,7,7)
        energy_stack = uniform_filter(energy_stack.astype(np.float32), size=size_block)
        
        # transpose to (z, y, x)
        energy_stack_transpose = np.transpose(energy_stack, (2, 1, 0))
        # print(energy_stack_transpose.shape)

        # divided by 8 to normalize
        dx = ndimage.sobel(energy_stack_transpose, axis=2)  / 8.0 # x = axis 2
        dy = ndimage.sobel(energy_stack_transpose, axis=1)  / 8.0 # y = axis 1
        dz = ndimage.sobel(energy_stack_transpose, axis=0)  / 8.0 # z = axis 0



        vectors = np.stack((dx, dy, dz), axis=-1)

        # show every 4 voxels
        step = 4  
        sub_vectors = vectors[::step, ::step, ::step]
        sub_shape = sub_vectors.shape[:3]

       
        z, y, x = np.meshgrid(
            np.arange(0, energy_stack_transpose.shape[0], step),
            np.arange(0, energy_stack_transpose.shape[1], step),
            np.arange(0, energy_stack_transpose.shape[2], step),
            indexing='ij'
        )
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)     # (N, 3)
        vecs = sub_vectors.reshape(-1, 3)                         # (N, 3)

        
        plotter = pv.Plotter()
        plotter.add_arrows(points, vecs, mag=1.5)  
        plotter.show()

    
                

def select_fingerprint_area(energy_img_list):
    energy_img_arr = np.array(energy_img_list)
    segment_list = []
    temp_segment_list = []
    max_energy = energy_img_arr.max()
    threshold = max_energy * 0.2
    for i in range(len(energy_img_list)):
        segment_energy = np.where(energy_img_list[i] > threshold, energy_img_list[i], 0).astype(float)
        temp_energy = np.where(energy_img_list[i] > threshold, 1, 0).astype(float)
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
    
    paired_energy_list = np.array(paired_energy_list)
    paired_segment_list = np.array(paired_segment_list)
    temp_segment_list = np.array(temp_segment_list)

    # transpose to (y, x, z) from (z, y, x)
    paired_energy_list = paired_energy_list.transpose(1, 2, 0)
    paired_segment_list = paired_segment_list.transpose(1, 2, 0)
    temp_segment_list = temp_segment_list.transpose(1, 2, 0)

    # print(paired_energy_list.shape)
    
    ############################################ 1 st method : using library ####################################################

    structure = ndimage.generate_binary_structure(3, 3)  # 18- หรือ 26-connectivity
    dilated_volume = ndimage.binary_dilation(paired_segment_list, structure=structure, iterations=1)

    and_vol = np.logical_and(dilated_volume, temp_segment_list)
    # plt.figure()
    # plt.imshow(temp_segment_list[:,:,3], cmap="gray")
    # plt.figure()
    # plt.imshow(dilated_volume[:,:,3], cmap="gray")
    # plt.figure()
    # plt.imshow(and_vol[:,:,3], cmap="gray")
    # plt.show()
    # print(dilated_volume.shape)

    
    labeled, num_obj = ndimage.label(and_vol, structure=structure)
    # filter only 2 largest components
    component_sizes = [(label_id, np.count_nonzero(labeled == label_id)) for label_id in range(1, num_obj + 1)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    candidate_components = set([label for label,_ in component_sizes[:4]])


    print("the number of connected components after dilation:", num_obj)
    filtered_mask = np.zeros_like(labeled, dtype=bool)
    plotter = pv.Plotter()
    for label_id in range(1, num_obj + 1):
        # comdition 1
        if label_id not in candidate_components:
            continue
        component = (labeled == label_id)
        # comditions 2
        if np.count_nonzero(component) < 500:   
            continue
        # comditions 3
        z_coords = np.where(component)[2]
        z_range = z_coords.max() - z_coords.min() + 1 if z_coords.size > 0 else 0
        if z_range < 3:
            continue
        
        filtered_mask |= component
        # mask = (labeled == label_id)
        # z_coords = np.where(mask)[2]  
        # z_range = z_coords.max() - z_coords.min() + 1 if z_coords.size > 0 else 0
        # print(f"Component {label_id}: z-range = {z_range} slices")


        # visualization
        verts, faces, _, _ = measure.marching_cubes(component, level=0.5, spacing=(1, 1, 10))
        faces = np.hstack([[3, *f] for f in faces])
        mesh = pv.PolyData(verts, faces)
        plotter.add_mesh(mesh, color=np.random.rand(3), opacity=0.6)
    plt.figure()
    plt.imshow(filtered_mask[:,:,0], cmap="gray")
    plotter.show()
    plt.show()
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


def applied_watershed(energy_img_list):

    blank = np.zeros_like(energy_img_list[0])


    for i in range(len(energy_img_list)):


        prev_energy = energy_img_list[(i-1)%8]
        curr_energy = energy_img_list[i]
        next_energy = energy_img_list[(i+1)%8]

        prev_max_energy = prev_energy.max()
        curr_max_energy = curr_energy.max()
        next_max_energy = next_energy.max()


        for j in range(10, 5, -1):

            prev_th = (j/10) * prev_max_energy
            curr_th = (j/10) * curr_max_energy
            next_th = (j/10) * next_max_energy

            max_th = max([prev_th, curr_th, next_th])

            prev_seg_max_energy = np.where(prev_energy > max_th, 1, 0).astype(np.uint8)   
            curr_seg_max_energy = np.where(curr_energy > max_th, 1, 0).astype(np.uint8)
            next_seg_max_energy = np.where(next_energy > max_th, 1, 0).astype(np.uint8)

            label_prev = label(prev_seg_max_energy)
            label_curr = label(curr_seg_max_energy)
            label_next = label(next_seg_max_energy)

            blank[prev_seg_max_energy == 1] = 1
            blank[curr_seg_max_energy == 1] = 1
            blank[next_seg_max_energy == 1] = 1

            largest_mask = select_largest_region(blank)

    return largest_mask

###########################  Path #########################
# input_file_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\049L3U.bmp"
raw_files_path = glob(r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\fillteredImg/" + "*")
input_files_path = glob(r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\sectoring_16_sectors/" + "*")
output_path = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\pixel_based_segment/"
output_path_2 = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\watershed_idea_segment/"
output_path_3 = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\spatial_energy\watershed_segment_fixed/"
temp_input = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\fillteredImg\080L8U.bmp.bmp"


raw_files_path.sort()
input_files_path.sort()

for idx in range(len(raw_files_path)):

    raw_img = cv.imread(raw_files_path[idx])
    raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    path = glob(input_files_path[idx] + "/" + "*")
    path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


    energy_img_list = []
    # each sectors
    for i in range(len(path)):

        input_img = cv.imread(path[i])
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

        block_size = 33

        energy = find_energy(gray_img, block_size=block_size)

        energy_img_list.append(energy)


    select_fingerprint_area(energy_img_list)


    ################## save files #########################
    base_filename = os.path.basename(raw_files_path[idx])
    output_file_name = output_path + base_filename
    # cv.imwrite(output_file_name, output_img)
    # plt.imsave(output_file_name, output_img)
    # io.imsave(output_file_name, output_img)

    base_filename = os.path.basename(raw_files_path[idx])
    output_file_name = output_path_3 + base_filename
    # cv.imwrite(output_file_name, new_output_img)
    # plt.imsave(output_file_name, new_output_img)
    # io.imsave(output_file_name, new_output_img)