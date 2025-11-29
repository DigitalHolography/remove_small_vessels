import numpy as np
from skimage.measure import label
from skimage.morphology import disk, skeletonize
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from skimage.segmentation import watershed, find_boundaries
import math

def disk_mask(numX, numY, R1, R2=math.sqrt(2), center=(0.5, 0.5)):
    """
    Creates a binary disk-shaped mask on a normalized grid.

    Parameters:
        numX (int): number of rows (height)
        numY (int): number of columns (width)
        R1 (float): inner radius
        R2 (float): outer radius (default 2)
        center (tuple): center of the disk in normalized coordinates (default (0.5, 0.5))

    Returns:
        mask (np.ndarray): binary mask of shape (numX, numY)
    """
    if R1 > R2:
        raise ValueError("R1 must be less than or equal to R2")

    x_center, y_center= center

    # normalized grid from 0 to 1
    x = np.linspace(0, 1, numX)
    y = np.linspace(0, 1, numY)
    X, Y = np.meshgrid(x, y)  # X = cols, Y = rows (MATLAB style)

    R = (X - x_center) ** 2 + (Y - y_center) ** 2

    if R2 == math.sqrt(2):  # only R1 was given (default R2)
        mask = R <= R1 ** 2
    else:
        mask = (R > R1 ** 2) & (R <= R2 ** 2)

    return mask.astype(bool)

def keep_components_connected_to_mask(vessel_mask, section_mask):
    """
    Keep only the connected components of `vessel_mask` that touch
    the 'section_mask'
    """

    # 1. Label vessel components
    labeled = label(vessel_mask, connectivity=2)  # 8-connected labeling

    # 2. Find labels that intersect with the squeleton
    labels_touching = np.unique(labeled[section_mask & (labeled > 0)])

    # 3. Build final mask (keep only the good labels)
    out = np.isin(labeled, labels_touching).astype(np.uint8)

    return out

def get_labeled_vesselness(mask, x_center, y_center, r1=0.15, circle_mask=None):
    # Skeletonize and remove central circle
    skel = skeletonize(mask)

    if circle_mask is None:
        numX, numY = mask.shape
        circle_mask = disk_mask(numX, numY, r1, center=(x_center / numX, y_center/ numY))
        
    skel = skel & ~circle_mask

    # Remove branch points
    neigh = np.array([[1,1,1],[1,10,1],[1,1,1]])
    bp_map = convolve(skel.astype(int), neigh, mode='constant') >= 13  # heuristic
    skel_no_branches = skel & ~binary_dilation(bp_map, disk(2))

    # Label branches
    label_skel = label(skel_no_branches)
    n = label_skel.max()

    # Distance transform (negative for watershed)
    D = -distance_transform_edt(mask)
    D[~(mask)] = -np.inf

    # Markers from skeleton
    markers = label_skel > 0
    markers = binary_dilation(markers, disk(1))

    # Watershed
    L = watershed(~markers)
    edges = find_boundaries(L, mode='outer')
        
    L[edges] = 0
    # L[L>1] = 1

    L = L * mask

    labeled_vessels = np.zeros_like(mask, dtype=int)
    for i in range(1, n + 1):
        branch_pixels = (L == i)
        labeled_vessels[branch_pixels] = i
    
    labeled_vessels *= ~circle_mask

    return labeled_vessels, edges