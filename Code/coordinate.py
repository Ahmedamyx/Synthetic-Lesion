import os
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, label as cc_label
from glob import glob
import matplotlib.pyplot as plt
 
import tkinter as tk
from tkinter import filedialog, messagebox
 
 
def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.affine
 
 
def compute_com(mask, label):
    """Compute center of mass for a given label."""
    return center_of_mass(mask == label)
 
 
def voxel_to_world(voxel, affine):
    """Convert voxel coordinates to world coordinates using affine."""
    voxel_hom = np.append(voxel, 1.0)  # [x, y, z, 1]
    world = affine @ voxel_hom
    return world[:3]
 
 
def gram_schmidt(vectors):
    """Orthonormalize vectors with Gram-Schmidt."""
    orthonormal = []
    for v in vectors:
        w = v - sum(np.dot(v, u) * u for u in orthonormal)
        if np.linalg.norm(w) > 1e-6:
            orthonormal.append(w / np.linalg.norm(w))
    return np.array(orthonormal)
 
 
def get_paths_with_tk():
    """Use Tkinter dialogs to select the derivatives and SynthSeg folders."""
    root = tk.Tk()
    root.withdraw()  # Hide main window
 
    messagebox.showinfo(
        "Select derivatives folder",
        "Please select the derivatives folder (containing sub-strokecase0001, etc.)"
    )
    derivatives_path = filedialog.askdirectory(title="Select derivatives folder")
    if not derivatives_path:
        messagebox.showerror("Error", "No derivatives folder selected. Aborting.")
        root.destroy()
        return None, None
 
    messagebox.showinfo(
        "Select SynthSeg output folder",
        "Please select the SynthSeg output folder (containing sub-strokecase0001, etc.)"
    )
    synthseg_path = filedialog.askdirectory(title="Select SynthSeg output folder")
    if not synthseg_path:
        messagebox.showerror("Error", "No SynthSeg folder selected. Aborting.")
        root.destroy()
        return None, None
 
    root.destroy()
    return derivatives_path, synthseg_path
 
 
def main():
    derivatives_path, synthseg_path = get_paths_with_tk()
    if derivatives_path is None or synthseg_path is None:
        print("No paths selected. Exiting.")
        return []
 
    results = []
 
    # For plotting at the end
    stroke_labels = []
    distances = []
 
    # Loop over all cases in derivatives
    for deriv_case_folder in sorted(glob(os.path.join(derivatives_path, "sub-*"))):
        case_name = os.path.basename(deriv_case_folder)
 
        ses_folder = os.path.join(deriv_case_folder, "ses-0001")
        lesion_mask_file = os.path.join(
            ses_folder,
            f"{case_name}_ses-0001_msk.nii.gz"
        )
 
        synthseg_file = os.path.join(
            synthseg_path,
            case_name,
            f"{case_name}_ses-0001_dwi_synthseg.nii.gz"
        )
 
        if not os.path.exists(lesion_mask_file) or not os.path.exists(synthseg_file):
            print(f"Skipping {case_name}, missing files")
            continue
 
        print(f"Processing {case_name}...")
 
        lesion_mask, lesion_affine = load_nifti(lesion_mask_file)
        synthseg_mask, synthseg_affine = load_nifti(synthseg_file)
 
        # --- Build atlas reference system from SynthSeg labels ---
 
        # Labels: adjust if needed (16=brainstem, 3=L cortex, 42=R cortex, 14=ventricle)
        brainstem_vox = compute_com(synthseg_mask, 16)
        lcortex_vox = compute_com(synthseg_mask, 3)
        rcortex_vox = compute_com(synthseg_mask, 42)
        ventricle_vox = compute_com(synthseg_mask, 14)
 
        # Convert to world
        brainstem_com = voxel_to_world(brainstem_vox, synthseg_affine)
        lcortex_com = voxel_to_world(lcortex_vox, synthseg_affine)
        rcortex_com = voxel_to_world(rcortex_vox, synthseg_affine)
        ventricle_com = voxel_to_world(ventricle_vox, synthseg_affine)
 
        # Skip case if any COM is invalid
        if np.any(np.isnan(brainstem_com)) or np.any(np.isnan(lcortex_com)) or \
           np.any(np.isnan(rcortex_com)) or np.any(np.isnan(ventricle_com)):
            print(f"Skipping {case_name}, invalid atlas COMs (NaNs).")
            continue
 
        # Define orthonormal axes (atlas system)
        axes = gram_schmidt([
            lcortex_com - brainstem_com,
            rcortex_com - brainstem_com,
            ventricle_com - brainstem_com
        ])
        origin = brainstem_com
 
        if axes.shape != (3, 3):
            print(f"Skipping {case_name}, Gram-Schmidt did not produce 3 axes.")
            continue
 
        # --- Find each stroke in the lesion mask as connected components ---
 
        # Binary lesion mask
        lesion_binary = lesion_mask > 0
 
        labeled_cc, num_cc = cc_label(lesion_binary)
 
        if num_cc == 0:
            print(f"No lesions found in {case_name}.")
            continue
 
        for idx in range(1, num_cc + 1):
            # Center of mass of this stroke (component)
            lesion_com_voxel = center_of_mass(lesion_binary, labeled_cc, idx)
            lesion_com_world = voxel_to_world(lesion_com_voxel, lesion_affine)
 
            # Lesion COM in atlas coordinates
            lesion_coord_atlas = np.dot(lesion_com_world - origin, axes.T)
            distance_to_origin = np.linalg.norm(lesion_coord_atlas)
 
            stroke_id = f"{case_name}_stroke{idx}"
 
            results.append({
                "case": case_name,
                "stroke_index": idx,
                "stroke_id": stroke_id,
                "lesion_com_voxel": lesion_com_voxel,
                "lesion_com_world": lesion_com_world,
                "lesion_coord_atlas": lesion_coord_atlas,
                "distance_to_origin": distance_to_origin
            })
 
            stroke_labels.append(stroke_id)
            distances.append(distance_to_origin)
 
    # --- Plot graph of distance of each stroke across whole dataset ---
 
    if len(distances) == 0:
        print("No strokes found in the dataset. Nothing to plot.")
        return results
 
 
    if len(distances) == 0:
        print("No strokes found in the dataset. Nothing to plot.")
        return results
 
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, edgecolor='black', alpha=0.7)
 
    plt.xlabel("Distance to atlas origin (mm)")
    plt.ylabel("Number of strokes")
    plt.title("Distribution of Stroke Distances in Atlas Space")
 
    # Optional: add mean/median lines
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    plt.axvline(mean_dist, color='red', linestyle='--', label=f"Mean = {mean_dist:.1f} mm")
    plt.axvline(median_dist, color='green', linestyle='--', label=f"Median = {median_dist:.1f} mm")
    plt.legend()
 
    plt.tight_layout()
    plt.show()
 
    print("Processing complete.")
    return results
 
 
if __name__ == "__main__":
    results = main()