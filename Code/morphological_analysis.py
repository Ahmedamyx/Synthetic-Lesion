import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog


# ---------------------
# Utility Functions
# ---------------------

def find_matching_cases(synthseg_dir, mask_dir):
    """Find matching subject IDs between SynthSeg outputs and stroke masks."""
    synthseg_path = Path(synthseg_dir)
    mask_path = Path(mask_dir)

    synthseg_cases = {p.name for p in synthseg_path.glob("sub-strokecase*")}
    mask_cases = {p.name for p in mask_path.glob("sub-strokecase*")}

    common_cases = sorted(synthseg_cases.intersection(mask_cases))
    cases = []
    for subject_name in common_cases:
        case_num = subject_name.replace("sub-strokecase", "")
        mask_file = mask_path / subject_name / "ses-0001" / f"{subject_name}_ses-0001_msk.nii.gz"
        synthseg_file = synthseg_path / subject_name / f"{subject_name}_ses-0001_dwi_synthseg.nii.gz"

        if mask_file.exists() and synthseg_file.exists():
            cases.append({
                "case_num": case_num,
                "subject_name": subject_name,
                "mask_path": mask_file,
                "synthseg_path": synthseg_file,
            })
    return cases


def find_connected_components(mask_data, min_size=10):
    """Find connected components (3D lesions)."""
    labeled, n = ndimage.label(mask_data > 0)
    sizes = ndimage.sum(mask_data, labeled, range(1, n + 1))
    valid = [i+1 for i, s in enumerate(sizes) if s >= min_size]
    filtered = np.isin(labeled, valid).astype(np.uint8)
    labeled, n = ndimage.label(filtered)
    return labeled, n


def compute_surface_voxels(binary_mask):
    """Estimate surface voxels count using binary erosion."""
    eroded = ndimage.binary_erosion(binary_mask)
    surface = binary_mask ^ eroded
    return np.sum(surface)


def compute_shape_descriptors(binary_mask):
    """Compute morphological descriptors in voxel space."""
    coords = np.argwhere(binary_mask)
    if len(coords) < 10:
        return None  # too small for reliable geometry
    
    volume = coords.shape[0]
    
    # --- Surface Area ---
    surface_voxels = compute_surface_voxels(binary_mask)
    surface_area = float(surface_voxels)

    # --- Convex Hull ---
    try:
        hull = ConvexHull(coords)
        convex_volume = hull.volume if hull.volume > 0 else np.nan
        convex_surface = hull.area if hull.area > 0 else np.nan
    except Exception:
        convex_volume, convex_surface = np.nan, np.nan

    # --- Eccentricity (from inertia tensor eigenvalues) ---
    cov = np.cov(coords.T)
    evals, _ = np.linalg.eig(cov)
    evals = np.sort(np.real(evals))
    if evals[-1] == 0:
        eccentricity = 0
    else:
        eccentricity = np.sqrt(1 - evals[0]/evals[-1])  # approximate for 3D

    # --- Compactness ---
    compactness = (volume ** 2) / (surface_area ** 3 + 1e-6)

    # --- Roundness (Sphericity) ---
    roundness = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / (surface_area + 1e-6)

    # --- Convexity ---
    convexity = volume / (convex_volume + 1e-6) if convex_volume > 0 else np.nan

    # --- Roughness ---
    roughness = surface_area / (convex_surface + 1e-6) if convex_surface > 0 else np.nan

    return {
        'Volume': volume,
        'Eccentricity': float(eccentricity),
        'Compactness': float(compactness),
        'Roundness': float(roundness),
        'Convexity': float(convexity),
        'Roughness': float(roughness)
    }


# ---------------------
# Main Analysis
# ---------------------

def compute_shape_statistics(synthseg_dir, mask_dir, min_stroke_size=10):
    cases = find_matching_cases(synthseg_dir, mask_dir)
    if not cases:
        print("⚠️ No matching subjects found. Check that sub-strokecase folders exist in both directories.")
        return pd.DataFrame()

    all_results = []
    for case in cases:
        print(f"Processing {case['subject_name']}...")
        mask_img = nib.load(case['mask_path'])
        mask_data = mask_img.get_fdata() > 0

        labeled, n_strokes = find_connected_components(mask_data, min_stroke_size)
        print(f"  Found {n_strokes} lesion(s)")

        for i in range(1, n_strokes + 1):
            lesion_mask = (labeled == i)
            desc = compute_shape_descriptors(lesion_mask)
            if desc:
                desc['Case'] = case['subject_name']
                desc['Stroke_ID'] = i
                all_results.append(desc)

    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv("stroke_shape_descriptors.csv", index=False)
        print(f"\n✅ Saved results to stroke_shape_descriptors.csv ({len(df)} strokes total)")
    else:
        print("\n⚠️ No strokes found above minimum size threshold.")
    return df


def plot_descriptor_distributions(df):
    """Plot distributions for each morphological descriptor."""
    if df.empty:
        print("⚠️ No data to plot.")
        return

    descriptors = ['Eccentricity', 'Compactness', 'Roundness', 'Convexity', 'Roughness']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, desc in enumerate(descriptors):
        ax = axes[i]
        valid = df[desc].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(valid, bins=30, density=True, alpha=0.6, color='steelblue')
        ax.set_title(desc)
        ax.set_xlabel(desc)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stroke_shape_descriptor_distributions.png", dpi=300)
    plt.show()
    print("✅ Saved plot as stroke_shape_descriptor_distributions.png")


# ---------------------
# Run Script
# ---------------------
if __name__ == "__main__":
    # Folder prompts
    Tk().withdraw()
    print("Please select your SynthSeg directory (contains sub-strokecase folders with *_synthseg.nii.gz)...")
    synthseg_dir = filedialog.askdirectory(title="Select SynthSeg directory")
    if not synthseg_dir:
        print("No SynthSeg directory selected. Exiting.")
        exit()

    print("Now select your Stroke Mask directory (contains sub-strokecase folders with *_msk.nii.gz)...")
    mask_dir = filedialog.askdirectory(title="Select Stroke Mask directory")
    if not mask_dir:
        print("No mask directory selected. Exiting.")
        exit()

    print(f"\n🧠 Selected directories:\n  SynthSeg: {synthseg_dir}\n  Masks: {mask_dir}\n")

    df = compute_shape_statistics(synthseg_dir, mask_dir, min_stroke_size=10)
    plot_descriptor_distributions(df)
