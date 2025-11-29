import numpy as np
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image

def make_colormap():
    """
    Recreates the MATLAB cmapLAB transitions.
    """
    artery_colors = [
        (0.00, (0, 0, 0)),   # black
        (1/3, (1, 0, 0)),    # red
        (2/3, (1, 1, 0)),    # yellow
        (1.00, (1, 1, 1)),   # white
    ]

    vein_colors = [
        (0.00, (0, 0, 0)),   # black
        (1/3, (0, 0, 1)),    # blue
        (2/3, (0, 1, 1)),    # cyan
        (1.00, (1, 1, 1)),   # white
    ]

    return LinearSegmentedColormap.from_list("cmap", artery_colors, N=256), LinearSegmentedColormap.from_list("cmap", vein_colors, N=256)

def apply_colormap_to_mask(M0, artery_mask, vein_mask, cmap_artery, cmap_vein):
    """
    M0   : grayscale uint8 image [0..255]
    mask : binary mask (0/1 or False/True)
    cmap : matplotlib colormap

    Returns an RGB image where:
    - inside mask → colormap applied to normalized M0
    - outside mask → grayscale in RGB
    """

    # --- Ensure correct dtypes ---
    M0 = M0.astype(np.float32)              # keep precision
    artery_mask = artery_mask.astype(bool)               # convert 0/1 → boolean
    vein_mask = vein_mask.astype(bool)               # convert 0/1 → boolean

    # --- Normalize M0 to [0,1] like MATLAB rescale() ---
    M0_norm = (M0 - M0.min()) / (np.ptp(M0) + 1e-8)

    # --- Base grayscale RGB ---
    base_rgb = np.dstack([M0_norm] * 3)

    # --- Colormap applied only inside mask ---
    colored_artery = cmap_artery(M0_norm)[..., :3]        # RGBA → RGB
    colored_vein = cmap_vein(M0_norm)[..., :3]        # RGBA → RGB

    # --- Merge ---
    vein = base_rgb.copy()
    artery = base_rgb.copy()
    av = base_rgb.copy()
    artery[artery_mask] = colored_artery[artery_mask]
    av[artery_mask] = colored_artery[artery_mask]
    vein[vein_mask] = colored_vein[vein_mask]
    av[vein_mask] = colored_vein[vein_mask]

    return artery, vein, av

def save_mask_image(img_rgb_float, out_path):
    """
    img_rgb_float : RGB float image [0..1]
    Saves as PNG uint8.
    """
    img_uint8 = (np.clip(img_rgb_float, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img_uint8).resize((1023,1023), Image.BILINEAR).save(out_path)
    # imageio.imwrite(out_path, img_uint8)

def generate_vessel_overlay(M0, mask_artery, mask_vein, artery_path, vein_path, av_path, vessel="artery"):
    """
    M0: grayscale uint8 image [0..255]
    mask: binary mask (0/1)
    out_path: path to .png file to save
    vessel: "artery" or "vein"
    """
    cmap_artery, cmap_vein = make_colormap()
    artery, vein, av = apply_colormap_to_mask(M0, mask_artery, mask_vein, cmap_artery, cmap_vein)
    save_mask_image(artery, artery_path)
    save_mask_image(vein, vein_path)
    save_mask_image(av, av_path)
