import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage.measure import regionprops, label
from scipy.ndimage import uniform_filter
from scipy import ndimage
import pyvista as pv
from skimage.color import label2rgb

from utils.Morphological_Module import fillHoles
from utils.Fourier_Module import Fourier2D

class TrackFingerPrint:

    def __init__(self, filtered_img, sectors_list):
        self.filtered_img = filtered_img
        self.sectors_list = sectors_list
        self.energy_img_list = []
        self.segments = []
        self.differences = []
        self.output_img = None
    


    ########################### Function ######################
    def _dilate3D(self, mask, radius):
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


    def _select_fingerprint_area(self, energy_img_list, th):
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
            # print(f"sector {i}")
            segment_energy = np.where(energy_img_arr_norm[i] > threshold, energy_img_arr_norm[i], 0).astype(float)
            temp_energy = np.where(energy_img_arr_norm[i] > threshold, 1, 0).astype(float)
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
        
        paired_energy_list = np.array(paired_energy_list)
        paired_segment_list = np.array(paired_segment_list)
        temp_segment_list = np.array(temp_segment_list)

        # transpose to (y, x, z) from (z, y, x)
        paired_energy_list = paired_energy_list.transpose(1, 2, 0)
        paired_segment_list = paired_segment_list.transpose(1, 2, 0)
        temp_segment_list = temp_segment_list.transpose(1, 2, 0)

        ################################################################################################
        radius = 1
        size = 3
        dilated_volume = self._dilate3D(paired_segment_list, radius=radius)
        and_vol = np.logical_and(dilated_volume, temp_segment_list)
        
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
            filtered_mask |= component

        n_sectors = 18
        for z in range(n_sectors):
            sum_segment += filtered_mask[:,:,z]

        return sum_segment

        ################################################################################################################

    def _gaussian_window_2d(self, shape, sigma):
        h, w = shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return gauss

    def display_grid_image(self, input_img, block_size):
        fig, ax = plt.subplots()
        ax.imshow(input_img, cmap='gray')
        grid_size = block_size
        ax.set_xticks(np.arange(-0.5, input_img.shape[1], grid_size))
        ax.set_yticks(np.arange(-0.5, input_img.shape[0], grid_size))
        ax.grid(color='red', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.show()

    def _find_energy(self, input_img, block_size):

        img = input_img.astype(np.float32)
        # avg each block
        mean = uniform_filter(img, size=block_size, mode='reflect')

        # avg squared each block
        mean_of_square = uniform_filter(img**2, size=block_size, mode='reflect')


        local_variance = mean_of_square - mean**2


        local_variance_display = cv.normalize(local_variance, None, 0, 255, cv.NORM_MINMAX)
        local_variance_display = local_variance_display.astype(np.uint8)


        return local_variance_display


    def _horizontal_mirror(self, segment):
        w = segment.shape[1]
        center = w // 2
        left = segment[:, :center]
        right = np.fliplr(left)

        return np.hstack([left, right])

    def _vertical_mirror(self, image):
        H = image.shape[0]
        center = H // 2
        top = image[:center, :]
        bottom = np.flipud(top)
        return np.vstack([top, bottom])

    def _mirror_segment(self, segment):
        h_mirror = self._horizontal_mirror(segment)
        full_mirror = self._vertical_mirror(h_mirror)
        return full_mirror

    def _select_largest_region(self, segment):
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

    def _compare_segments(self, segment1, segment2):

        segment1_area = np.count_nonzero(segment1)
        # print(f"area 1 {segment1_area}")

        segment2_area = np.count_nonzero(segment2)
        # print(f"area 2 {segment2_area}")

        diff = np.abs(segment2_area - segment1_area)
        # print(f"diff {diff}")

        return diff

    def _normalize_feature(self, values, eps=1e-8):
        min_val = np.min(values)
        max_val = np.max(values)
        return [(v - min_val) / (max_val - min_val + eps) for v in values]

    def _find_freq(self, img):
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


    def _select_fingerprint_region(self, segment, input_img):
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

            variance = np.var(mask_img)
            variance_ratio = variance / (curr_label.area)
            compactness = (perimeter ** 2) / area

            minr, minc, maxr, maxc = curr_label.bbox
            height = maxr - minr
            width = maxc - minc
            bbox_sizes = [height, width]
            elongation_ratio = max(bbox_sizes) / min(bbox_sizes)

            variances_list.append(variance)
            elongation_list.append(elongation_ratio)
            label_list.append(curr_label)


        norm_variance = self._normalize_feature(variances_list)
        norm_elongation = self._normalize_feature(elongation_list)
        norm_elongation = [1 - c for c in norm_elongation]
        
        scores = []
        for i in range(len(variances_list)):

            score = norm_variance[i] + norm_elongation[i]
            scores.append((score, label_list[i]))

        scores.sort(key= lambda x: x[0], reverse=True)
        top_score, fingerprint_label = scores[0]

        fingerprint_mask[labeled_mask == fingerprint_label.label] = 1
        
        return fingerprint_mask
    
    def get_output_img(self):
        return self.output_img
    
    def _MSER(self, energy_img_list):
        mser = cv.MSER_create()

        binary_masks = []

        for z in range(len(energy_img_list)):
            img_slice = energy_img_list[z] 

            # print(f"sector {z}")


            # Detect regions
            regions, _ = mser.detectRegions(img_slice)

            mask = np.zeros_like(img_slice, dtype=np.uint8)

            for region in regions:
                cv.fillPoly(mask, [region.reshape(-1, 1, 2)], 255)

            # plt.imshow(mask, cmap="gray")
            # plt.show()

            binary_masks.append(mask)



        radius = 1
        size = 3
        binary_masks = np.array(binary_masks)

        binary_masks = binary_masks.transpose(1, 2, 0)
        dilated_volume = self._dilate3D(binary_masks, radius=radius)
        size = 3
        structure = np.ones((size, size, size), dtype=bool)
        labeled, num_obj = ndimage.label(dilated_volume, structure=structure)
        # print(num_obj)

        filtered_mask = np.zeros_like(labeled, dtype=bool)
        sum_segment = np.zeros_like(filtered_mask[:,:,0], dtype=bool)

        for label_id in range(1, num_obj + 1):
            component = (labeled == label_id)
            filtered_mask |= component

        n_sectors = 18
        for z in range(n_sectors):
            sum_segment += filtered_mask[:,:,z]
        

        return sum_segment



    
    def forward(self):
        for i in range(len(self.sectors_list)):
            block_size = 33
            energy = self._find_energy(self.sectors_list[i], block_size=block_size)
            self.energy_img_list.append(energy)
        for th in np.arange(0.7, 0.0, -0.05):
            segment = self._select_fingerprint_area(self.energy_img_list, th=th)
            self.segments.append(segment)
        
        # mser_segment = self._MSER(self.energy_img_list)
        # label_img = label(mser_segment)
        # rgb_img = label2rgb(label_img, bg_label=0)

        # plt.imshow(rgb_img)
        # plt.show()

        n = len(self.segments)

        for j in range(n-1):
            # print(j)
            seg1 = self.segments[j]
            seg2 = self.segments[(j + 1)]  
            diff = self._compare_segments(seg1, seg2)  
            self.differences.append(diff)
        
        max_diff = max(self.differences)
        idx_max_diff = self.differences.index(max_diff)
        print(f"threshold = {round(0.7 - idx_max_diff * 0.05, 2)}")
        selected_segment = self.segments[idx_max_diff]
        fillhole_segment = fillHoles(selected_segment.astype(np.float32))
        fingerprint_segment = self._select_fingerprint_region(fillhole_segment, self.filtered_img)
        self.output_img = self.filtered_img * fingerprint_segment
        self.output_img[fingerprint_segment == 0] = self.filtered_img.mean()

        # fillhole_segment = fillHoles(mser_segment.astype(np.float32))
        # fingerprint_segment = self._select_fingerprint_region(fillhole_segment, self.filtered_img)
        # self.output_img = self.filtered_img * fingerprint_segment
        # self.output_img[fingerprint_segment == 0] = self.filtered_img.mean()
        

    
        

