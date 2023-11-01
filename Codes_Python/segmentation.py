import KTCScoring
from skimage.segmentation import chan_vese
from scipy import ndimage
import numpy as np


def cv(deltareco_pixgrid):
    mu = np.mean(deltareco_pixgrid)
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(abs(deltareco_pixgrid), mu=0.1, lambda1=1, lambda2=1, tol=1e-6,
                max_num_iter=1000, dt=2.5, init_level_set="checkerboard",
                extended_output=True)

    labeled_array, num_features = ndimage.label(cv[0])
    # Initialize a list to store masks for each region
    region_masks = []

    # Loop through each labeled region
    deltareco_pixgrid_segmented = np.zeros((256,256))

    for label in range(1, num_features + 1):
        # Create a mask for the current region
        region_mask = labeled_array == label
        region_masks.append(region_mask)
        if np.mean(deltareco_pixgrid[region_mask]) < mu:
            deltareco_pixgrid_segmented[region_mask] = 1
        else:
            deltareco_pixgrid_segmented[region_mask] = 2

    return deltareco_pixgrid_segmented


def otsu(deltareco_pixgrid):
    level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)
    deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)

    ind0 = deltareco_pixgrid < x[level[0]]
    ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]],deltareco_pixgrid <= x[level[1]])
    ind2 = deltareco_pixgrid > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class

    match bgclass:
        case 0:
            deltareco_pixgrid_segmented[ind1] = 2
            deltareco_pixgrid_segmented[ind2] = 2
        case 1:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind2] = 2
        case 2:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind1] = 1

    return deltareco_pixgrid_segmented