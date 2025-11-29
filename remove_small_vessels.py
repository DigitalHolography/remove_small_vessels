#!/bin/env python

import os
import sys
import glob
import numpy as np
import argparse
from PIL import Image
import model_utils
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import graph_struct.skeleton_graph as skg
import graph_struct.sknw as sknw
import mask_utils
import eyeflow_utils
from typing import Any, List, Optional
import shutil
from huggingface_hub import hf_hub_download, hf_hub_url
import requests

def get_branch_segments(skel):
    """Break skeleton into segments by removing junction pixels."""
    h, w = skel.shape
    
    # compute degree (count neighbors)
    deg = np.zeros_like(skel, dtype=np.uint8)
    for y in range(1,h-1):
        for x in range(1,w-1):
            if skel[y,x]:
                deg[y,x] = skel[y-1:y+2, x-1:x+2].sum() - 1
    
    # junction = degree >= 3
    junctions = (deg >= 3)
    
    # remove junctions to separate segments
    skel_no_junctions = skel.copy()
    skel_no_junctions[junctions] = 0
    
    # label segments
    segments = label(skel_no_junctions, connectivity=2)
    
    return segments, junctions


def prune_skeleton_by_radius(mask, radius_thresh=None):
    """Keep only skeleton branches whose radius >= threshold."""
    skel = skeletonize(mask)
    edt = distance_transform_edt(mask)

    segments, junctions = get_branch_segments(skel)

    out = np.zeros_like(skel)
    # add junctions back
    out[junctions] = 1

    radii_list=[]

    radius_max = 0
    l=[]

    for seg_id in np.unique(segments):
        if seg_id == 0:
            continue
        
        coords = np.where(segments == seg_id)
        radii = edt[coords]
        l.append((coords, radii.mean()))
        if radius_thresh is not None: 
            if radii.mean() >= radius_thresh:
                out[coords] = 1
                radii_list.append(radii.mean())
        else:
            if radii.mean() > radius_max:
                radius_max = radii.mean()
             
    if radius_thresh is None:
        l.sort(key=lambda x: x[1], reverse=True)

        for coords, r in l:
            if r >= 0.6*radius_max:
                out[coords]=1
                radii_list.append(r)
            else:
                break

    return out, l


def reconstruct_trunks_via_traversal(full_mask, full_skel, large_skel, optic_center):
    """Reconstruct the large vessels, by exploring the full vessel graph, and keeping the largest segment at each intersection"""

    large_graph = skg.SkeletonGraph(sknw.build_sknw(large_skel))

    circle_mask = mask_utils.disk_mask(full_mask.shape[0], full_mask.shape[1], 0.1, center=(optic_center[0] / full_mask.shape[1], optic_center[1] / full_mask.shape[0]))
    full_skel = full_skel & ~circle_mask
    large_segments = label(large_skel & ~circle_mask, connectivity=2)
    large_graph_list = [skg.SkeletonGraph(sknw.build_sknw(large_segments==i)) for i in np.unique(large_segments)[1:]]

    # Build the graph to explore. It prolongates the vessels already selected, therefore they are removed from the graph.
    end_points = large_graph.endpoints
    for _, ep in end_points:
        large_graph.remove_node(ep)

    edt = distance_transform_edt(full_mask)
    full_graph = skg.SkeletonGraph(sknw.build_sknw(full_skel), edt)
    exploratory_graph = full_graph - large_graph

    for G_large in large_graph_list:
        optic_disc_ep = skg.get_closest_endpoint(G_large, optic_center)
        if optic_disc_ep is None:
            # no endpoint found
            optic_disc_ep = G_large.get_closest_node(optic_center)
        start_point = exploratory_graph.get_node(optic_disc_ep['o'][1], optic_disc_ep['o'][0])
        if start_point is None:
            # found no node corresponding
            continue
        large_skel, exploratory_graph = skg.explore_graph(large_skel, exploratory_graph, start_point[1])

    return large_skel


def keep_main_vessels(vessel_mask, x_center, y_center, radius_thresh=None):
    """
    Keep the main vessels of a mask. Starts by selecting the largest vessels (vessels with radius >= 0.6*max radius, or above a certain threshold)
    Then, reconstruct the main vessels through graph traversal.
    """
    # Skeletonize and prune by radius
    large_skel, l = prune_skeleton_by_radius(vessel_mask, radius_thresh)

    # Reconstruct remaining vessels
    vessel_skel = skeletonize(vessel_mask)
    large_skel = reconstruct_trunks_via_traversal(vessel_mask, vessel_skel, large_skel, (x_center, y_center))

    # Label vessel, by skeletonizing and removing intersections
    numX, numY = vessel_mask.shape
    circle_mask = mask_utils.disk_mask(numX, numY, 0.08, center=(x_center / numX, y_center/ numY))
    labeled_vessels, edges = mask_utils.get_labeled_vesselness(vessel_mask, None, None, None, circle_mask=circle_mask)

    # Keep only components touching the pruned skeleton
    main_vessels_sections = mask_utils.keep_components_connected_to_mask(labeled_vessels, (large_skel>0))

    # Add back edges
    main_vessels = main_vessels_sections.astype(bool) | ((edges & ~circle_mask) & vessel_mask.astype(bool))

    # Keep only components touching a larger circle around the optic disc
    outer_circle_mask = mask_utils.disk_mask(numX, numY, 0.15, center=(x_center / numX, y_center/ numY))
    main_vessels = mask_utils.keep_components_connected_to_mask(main_vessels, outer_circle_mask)

    return main_vessels

def find_removal_threshold(vessel_mask, x_center, y_center, radius_thresh, floor, ceil):
    """Fine tune the threshold used to select the main vessels"""

    # First try the given threshold
    main_vessels = keep_main_vessels(vessel_mask, x_center, y_center, radius_thresh=radius_thresh)

    nb_components = len(np.unique(label(main_vessels)))-1

    # If there is too much components, we need to augment the radius threshold
    if nb_components > 5:
        floor = radius_thresh
        # If we suddenly go from too much components to not enough, we prefer having too much
        if ceil is not None and (ceil - floor < 0.1):
            return main_vessels
        return find_removal_threshold(vessel_mask, x_center, y_center, radius_thresh=floor+1 if ceil is None else (floor + ceil)/2, floor=floor, ceil=ceil)

    # If there is not enough components, we need to lower the radius threshold
    if nb_components < 3:
        ceil = radius_thresh
        # If the upper bound is null, it must mean the mask only had one component. In that case, use radius=3
        if ceil <=0:
            return keep_main_vessels(vessel_mask, x_center, y_center, radius_thresh=3)
        ceil = radius_thresh
        return find_removal_threshold(vessel_mask, x_center, y_center, radius_thresh=ceil-1 if floor is None else (floor + ceil)/2, floor=floor, ceil=ceil)

    return main_vessels

def prune_small_vessels(vessel_mask, x_center, y_center, radius_thresh=None):
    """Prune small vessels using a given threshold. If there is too much/not enough vessels in the obtained mask, search for optimal threshold"""

    main_vessels = keep_main_vessels(vessel_mask, x_center, y_center, radius_thresh=radius_thresh)

    nb_components = len(np.unique(label(main_vessels)))-1

    is_floor = nb_components > 5
    is_ceil = nb_components < 2

    if radius_thresh is None:
        radius_thresh = 3
    elif is_floor:
        radius_thresh += 1
    else:
        radius_thresh -= 1

    return find_removal_threshold(vessel_mask, x_center, y_center, radius_thresh=3, floor=radius_thresh if is_floor else None, ceil=radius_thresh if is_ceil else None)


def select_highest_eyeflow_subdirectory(base_path: str):
    "Select eyeflow diretory with highest processing number (last generated)"

    # Filter entries to include only subdirectories
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Use a regex to extract the digit at the end of each subdirectory name
    digit_subdirs = []
    for subdir in subdirs:
        parts = subdir.split('_')
        if parts[-1].isdigit():
            digit_subdirs.append((subdir, int(parts[-1])))
    
    # Find the subdirectory containing pulsewave masks with the maximum digit
    while len(digit_subdirs) > 0:
        max_subdir = max(digit_subdirs, key=lambda x: x[1])
        max_subdir_path = os.path.join(base_path, max_subdir[0])

        pulsewave_masks_path = os.path.join(max_subdir_path, "png", "mask")
        if os.path.exists(pulsewave_masks_path):
            return max_subdir_path
        digit_subdirs.remove(max_subdir)
    return None


def create_large_vessel_masks(measure_folder_path: str, optic_disc_detector_path:str, radius_thresh:float=None):
    """Retrieves M0 image and artery/vein masks. Find optic disc center using the optic_disc_detector. Then create large vessel masks in folder. Previous masks are kept, with the suffix '_full'"""

    eyeflow_path = os.path.join(select_highest_eyeflow_subdirectory(os.path.join(measure_folder_path, "eyeflow")))
    if eyeflow_path is None or not os.path.exists(eyeflow_path):
        raise Exception("No eyeflow folder found")
    
    eyeflow_dir_name = os.path.basename(eyeflow_path)
    M0_path = os.path.join(eyeflow_path, "png", "mask", "steps", (eyeflow_dir_name + "_all_10_M0.png"))
    if not os.path.exists(M0_path):
        raise Exception("No M0.png found")
    
    mask_vein_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_maskVein.png"))
    mask_artery_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_maskArtery.png"))
    if not os.path.exists(mask_artery_path) or not os.path.exists(mask_vein_path):
        raise Exception("artery/vein mask path incorrect")
    
    mask_vein_full_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_maskVein_full.png"))
    mask_artery_full_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_maskArtery_full.png"))
    artery_full = os.path.exists(mask_artery_full_path)
    vein_full = os.path.exists(mask_vein_full_path)

    M0 = np.array(Image.open(M0_path).convert('L').resize((512,512), Image.BILINEAR))
    mask_vein = np.array(Image.open(mask_vein_path if not vein_full else mask_vein_full_path).resize((512,512), Image.NEAREST)).astype(np.uint8)
    mask_artery = np.array(Image.open(mask_artery_path if not artery_full else mask_artery_full_path).resize((512,512), Image.NEAREST)).astype(np.uint8)

    papilla_sess = model_utils.load_onnx_model(optic_disc_detector_path)
    x_center, y_center, _, _ = model_utils.get_bounding_box(M0, papilla_sess)

    mask_vein_large = prune_small_vessels(mask_vein, x_center, y_center, radius_thresh=radius_thresh)
    mask_artery_large = prune_small_vessels(mask_artery, x_center, y_center, radius_thresh=radius_thresh)

    if not os.path.exists(mask_vein_full_path):
        os.rename(mask_vein_path, mask_vein_full_path)
    if not os.path.exists(mask_artery_full_path):
        os.rename(mask_artery_path, mask_artery_full_path)

    Image.fromarray(mask_vein_large.astype(np.uint8)*255).resize((1023,1023), Image.NEAREST).save(mask_vein_path)
    Image.fromarray(mask_artery_large.astype(np.uint8)*255).resize((1023,1023), Image.NEAREST).save(mask_artery_path)

    overlay_vein_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_Vein.png"))
    overlay_artery_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_Artery.png"))
    overlay_av_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_RGB.png"))
    overlay_vein_full_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_Vein_full.png"))
    overlay_artery_full_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_Artery_full.png"))
    overlay_av_full_path = os.path.join(eyeflow_path, "png", "mask", (eyeflow_dir_name + "_M0_RGB_full.png"))

    if not os.path.exists(overlay_vein_full_path):
        os.rename(overlay_vein_path, overlay_vein_full_path)
    if not os.path.exists(overlay_artery_full_path):
        os.rename(overlay_artery_path, overlay_artery_full_path)
    if not os.path.exists(overlay_av_full_path):
        os.rename(overlay_av_path, overlay_av_full_path)
    
    eyeflow_utils.generate_vessel_overlay(M0=M0, mask_vein=mask_vein_large, mask_artery=mask_artery_large, artery_path=overlay_artery_path, vein_path=overlay_vein_path, av_path=overlay_av_path)

    print(f"Large masks generated for {measure_folder_path}")
    


def apply_script_to_each_measure(date_path: str, optic_disc_detector_path: str, revert:bool=False):
    """Applies script (either thin vessel removal or reinstauration of previous masks) to each measure in the current folder"""

    for measure_dir in os.listdir(date_path):		# either directly the correct subdir, or a subdir with just the name if several person were measured the same day
        name_path = os.path.join(date_path, measure_dir)
        if not os.path.isdir(name_path):
            continue

        parts = measure_dir.split('_')
        if '' in parts:
            parts.remove('')

        if len(parts) < 5:	 # if its only a name subdir, search the subdir containing the measures by recursively calling the func
            apply_script_to_each_measure(name_path, optic_disc_detector_path=optic_disc_detector_path, revert=revert)
        else:
            if revert:
                revert_masks_in_folder(name_path)
            else:
                try:
                    create_large_vessel_masks(name_path, optic_disc_detector_path=optic_disc_detector_path)
                except:
                    print(f"Script failed on {name_path}")


def apply_script_to_dataset(current_path: str, optic_disc_detector_path: str, revert:bool=False):
    """Find each folder of measures in the dataset"""
    for date_dir in os.listdir(current_path):		# date_dir since it usually is a date
        if len(date_dir) < 2 or date_dir[:2] != '25':
            continue

        date_path = os.path.join(current_path, date_dir)
        if not os.path.isdir(date_path) or date_dir == "#recycle":
            continue
        
        print(f"\n[Dataset] Applying script for each measure in {date_path}")
        apply_script_to_each_measure(date_path=date_path, optic_disc_detector_path=optic_disc_detector_path, revert=revert)


def revert_masks_in_folder(folder_path):
    """
    Renames all files ending with '_full.<ext>' by removing the '_full' suffix.
    """
    eyeflow_path = os.path.join(folder_path, "eyeflow")
    if not os.path.exists(eyeflow_path):
        return
    
    eyeflow_path = os.path.join(select_highest_eyeflow_subdirectory(eyeflow_path))
    mask_path = os.path.join(eyeflow_path, 'png', 'mask')
    if mask_path is None or not os.path.exists(mask_path):
        raise Exception("No eyeflow folder found")
    
    # Match files that end exactly with `_full.<ext>`
    pattern = os.path.join(mask_path, "*_full.*")
    files = glob.glob(pattern)

    for filepath in files:
        dirname, filename = os.path.split(filepath)

        # Split into base and extension
        base, ext = os.path.splitext(filename)

        # Only rename if base ends with _full
        if not base.endswith("_full"):
            continue

        new_base = base[:-5]   # remove "_full" (5 chars)
        new_name = new_base + ext
        new_path = os.path.join(dirname, new_name)

        os.rename(filepath, new_path)

    print(f"Large masks removed in {folder_path}")



def ensure_opticdisc_model(
    model_path: str,
    hf_repo: str = "your-username/your-repo",
    hf_filename: str = "opticdisc.onnx",
    repo_type: str = "model",            # "model" or "dataset"
    min_bytes: int = 10_000,            # minimal expected size in bytes (tweak)
    max_retries: int = 2,
) -> str:
    """
    Ensure the ONNX model exists at model_path. If missing or suspiciously small,
    try to download it from Hugging Face.

    - hf_repo: repo id like "username/repo" or "org/repo"
    - hf_filename: file path inside the repo (can include subfolders)
    - repo_type: usually "model", but set "dataset" if the file is in a dataset repo
    - min_bytes: heuristic threshold — adjust to expected model size
    - use_auth_token: token string for private repos (or None -> environment/autologin)
    """

    # If already exists and seems OK, return immediately
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size >= min_bytes:
            print(f"[Model] Found local model at {model_path} ({size} bytes).")
            return model_path
        else:
            print(
                f"[Model] Local model exists but is small ({size} bytes). Will re-download."
            )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Try hf_hub_download first (recommended)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Model] Attempt {attempt}: downloading via hf_hub_download()...")
            cached_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                repo_type=repo_type,
            )
            # If hf_hub_download succeeded, move/replace to target path
            if os.path.exists(cached_path):
                # If cached_path and model_path are same, fine
                if os.path.abspath(cached_path) != os.path.abspath(model_path):
                    shutil.copyfile(cached_path, model_path)
                size = os.path.getsize(model_path)
                print(f"[Model] Downloaded to {model_path} ({size} bytes).")
                # Quick sanity check on size
                if size < min_bytes:
                    raise ValueError(
                        f"Downloaded file too small ({size} bytes) — probably not the real model."
                    )
                return model_path
        except Exception as e:
            last_exc = e
            print(f"[Model][Warning] hf_hub_download attempt failed: {e}")

    # Fallback: direct HTTP GET from huggingface.co (raw file via /resolve/main/)
    try:
        print("[Model] Fallback: streaming download from huggingface.co URL...")
        # Build URL: https://huggingface.co/<repo>/resolve/main/<hf_filename>
        url = hf_hub_url(repo_id=hf_repo, filename=hf_filename, repo_type=repo_type)
        # hf_hub_url gives a URL; now stream it
        headers = {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            tmp_path = model_path + ".download"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        final_size = os.path.getsize(tmp_path)
        if final_size < min_bytes:
            # keep the downloaded file for inspection, but raise
            print(f"[Model][Error] Fallback download too small ({final_size} bytes).")
            raise ValueError("Downloaded file is suspiciously small.")
        # Move into place
        os.replace(tmp_path, model_path)
        print(f"[Model] Successfully saved model to {model_path} ({final_size} bytes).")
        return model_path
    except Exception as e:
        print(f"[Model][Error] Fallback HTTP download failed: {e}")
        if last_exc is not None:
            raise RuntimeError(
                f"Failed to download model via hf_hub_download ({last_exc}) and fallback HTTP ({e})."
            ) from e
        raise


def main(arguments: Optional[List[Any]] = None):
    """Main entry point for the script."""

    parser = argparse.ArgumentParser(description="Generate or revert large masks.")

    # Positional OR --folder for single-folder mode
    parser.add_argument(
        "folder",
        nargs="?",
        type=str,
        help="Path to a single measure folder (default mode)."
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Explicitly specify a single measure folder."
    )

    # Dataset mode
    parser.add_argument(
        "--dataset",
        type=str,
        help="Run on all measure folders contained in a dataset."
    )

    # Revert option
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert previously generated masks."
    )

    args = parser.parse_args(arguments)

    # Determine mode
    measure_folder = args.folder or args.folder
    dataset_folder = args.dataset
    revert_flag = args.revert

    optic_disc_detector_path = ensure_opticdisc_model(
    "model/opticdisc.onnx",
    hf_repo="DigitalHolography/EyeFlow_OpticDiscDetectorV2",
    hf_filename="opticdisc.onnx",
    repo_type="model",
    min_bytes=100000  # set to expected ONNX size
)

    if measure_folder is None and dataset_folder is None:
        parser.error("You must specify either a folder or a dataset.")

    if measure_folder and dataset_folder:
        parser.error("Choose ONLY one: a single folder OR a dataset.")

    # Run on single folder
    if measure_folder is not None:
        if revert_flag:
            print(f"[Single Folder] Reverting masks in folder: {measure_folder}")
            revert_masks_in_folder(measure_folder)
        else:
            print(f"[Single Folder] Generating masks in folder: {measure_folder}")
            create_large_vessel_masks(measure_folder, optic_disc_detector_path)
        return

    # Run on dataset
    if dataset_folder is not None:
        if revert_flag:
            print(f"[Dataset] Reverting masks in dataset: {dataset_folder}")
            apply_script_to_dataset(
                dataset_folder,
                optic_disc_detector_path,
                revert=True
            )
        else:
            print(f"[Dataset] Generating masks in dataset: {dataset_folder}")
            apply_script_to_dataset(
                dataset_folder,
                optic_disc_detector_path,
                revert=False
            )
        return

if __name__ == "__main__":
    main(sys.argv[1:])