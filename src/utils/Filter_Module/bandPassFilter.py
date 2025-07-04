import numpy as np

def distanceMap(size, pos):
    map_height = size[0]
    map_width = size[1]
    
    v = np.arange(0.5*((map_height+1)%2),
                  0.5*((map_height+1)%2)+map_height)
    
    u = np.arange(0.5*((map_height+1)%2),
                  0.5*((map_height+1)%2)+map_width)
    
    uu, vv = np.meshgrid(u, v)
    
    distance_map = ((uu-pos[1])**2 + (vv-pos[0])**2)**0.5
    
    return distance_map

def idealFunction(distance_map, band_center, band_width):
    ideal_func = distance_map.copy()
    ideal_func = np.where(
        (distance_map >= (band_center-band_width/2)) & (distance_map <= (band_center+band_width/2)),
        1, 0
    )
    
    return ideal_func

def gaussianFunction(distance_map, band_center, band_width):
    gauss_func = np.exp(-((distance_map**2-band_center**2) / (distance_map*band_width))**2)
    return gauss_func

def butterworthFunction(distance_map, band_center, band_width, n_order):
    bw_func = 1 / (1 + ((distance_map**2-band_center**2) / (distance_map*band_width))**(2*n_order))


def bandpassFilter(filter_size, filter_pos, band_center, band_width, filter_func, n_order= 2):
    
    distance_map = distanceMap(filter_size, filter_pos)
    
    filterFunction = {
        "Ideal" : idealFunction(distance_map, band_center, band_width),
        "Gaussian" : gaussianFunction(distance_map, band_center, band_width),
        "Butterworth" : butterworthFunction(distance_map, band_center, band_width, n_order)
    }
    
    bp_filter = filterFunction[filter_func]
    return bp_filter