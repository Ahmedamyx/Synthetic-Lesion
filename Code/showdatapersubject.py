import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_dilation


# ------------------------------------------------------------
# ANALYSIS FUNCTIONS (REUSED FROM YOUR CODE, LIGHTLY CLEANED)
# ------------------------------------------------------------

def analyze_mask(mask_img, original_img, adc_img, min_voxels=5, dilation_size=5):
    mask_data = mask_img.get_fdata()
    orig_data = original_img.get_fdata()

    # Normalize ADC if needed
    if adc_img is not None:
        adc_raw = adc_img.get_fdata()
        adc_data = adc_raw / 1000.0 if adc_raw.max() > 10 else adc_raw
    else:
        adc_data = None

    lesion_mask = (mask_data > 0)
    voxel_vol = np.prod(mask_img.header.get_zooms())

    num_voxels = np.sum(lesion_mask)
    volume_mm3 = num_voxels * voxel_vol
    volume_cm3 = volume_mm3 / 1000.0

    structure = np.ones((3,3,3), dtype=np.uint8)
    labeled_mask, num_lesions = label(lesion_mask, structure)

    lesion_sizes = np.bincount(labeled_mask.flatten())[1:]

    # Filter lesions by size
    filtered_labels = [i+1 for i, size in enumerate(lesion_sizes) if size >= min_voxels]
    filtered_mask = np.isin(labeled_mask, filtered_labels)

    filtered_lesion_volumes = []
    for label_idx in filtered_labels:
        size_vox = lesion_sizes[label_idx - 1]
        filtered_lesion_volumes.append(size_vox * voxel_vol / 1000.0)

    mean_intensity_filtered = float(orig_data[filtered_mask].mean()) if filtered_mask.any() else 0.0
    mean_intensity_adc = float(adc_data[filtered_mask].mean()) if (adc_data is not None and filtered_mask.any()) else 0.0

    # Surrounding tissue
    if filtered_mask.any():
        struct_elem = np.ones((dilation_size,)*3)
        dilated_mask = binary_dilation(filtered_mask, structure=struct_elem)
        surrounding = dilated_mask & (~filtered_mask)

        if surrounding.any():
            mean_intensity_surround = float(orig_data[surrounding].mean())
            mean_intensity_surround_adc = float(adc_data[surrounding].mean()) if adc_data is not None else 0.0
        else:
            mean_intensity_surround = 0.0
            mean_intensity_surround_adc = 0.0
    else:
        mean_intensity_surround = 0.0
        mean_intensity_surround_adc = 0.0

    return {
        "volume_cm3": volume_cm3,
        "num_voxels": int(num_voxels),
        "num_lesions_total": int(num_lesions),
        "num_lesions_filtered": len(filtered_labels),
        "mean_intensity_filtered": mean_intensity_filtered,
        "mean_intensity_adc": mean_intensity_adc,
        "mean_intensity_surrounding": mean_intensity_surround,
        "mean_intensity_surrounding_adc": mean_intensity_surround_adc,
        "filtered_lesion_volumes": filtered_lesion_volumes
    }


def find_image_pairs(root_dir):
    pairs = []
    derivatives = os.path.join(root_dir, "derivatives")

    for case in os.listdir(derivatives):
        case_dir = os.path.join(derivatives, case, "ses-0001")
        if not os.path.isdir(case_dir):
            continue

        mask_file = os.path.join(case_dir, f"{case}_ses-0001_msk.nii.gz")
        if not os.path.exists(mask_file):
            continue

        orig_file = os.path.join(root_dir, case, "ses-0001", "dwi", f"{case}_ses-0001_dwi.nii.gz")
        if not os.path.exists(orig_file):
            continue

        adc_file = os.path.join(root_dir, case, "ses-0001", "dwi", f"{case}_ses-0001_adc.nii.gz")
        if not os.path.exists(adc_file):
            adc_file = None

        pairs.append((case, orig_file, mask_file, adc_file))

    return pairs


# ------------------------------------------------------------
# GUI PART (CASE SELECTION + DISPLAY)
# ------------------------------------------------------------

def show_case_details(stats):
    """Create a new Tkinter window showing metrics for one case."""
    win = tk.Toplevel()
    win.title(f"Case Details: {stats['case_id']}")
    win.geometry("450x500")

    text = tk.Text(win, font=("Arial", 12), wrap=tk.WORD)
    text.pack(fill=tk.BOTH, expand=True)

    text.insert(tk.END, f"Case ID: {stats['case_id']}\n")
    text.insert(tk.END, "-"*40 + "\n\n")

    text.insert(tk.END, f"Total Lesion Volume: {stats['volume_cm3']:.4f} cm³\n")
    text.insert(tk.END, f"Total Voxels: {stats['num_voxels']}\n")
    text.insert(tk.END, f"Connected Lesions (Total): {stats['num_lesions_total']}\n")
    text.insert(tk.END, f"Connected Lesions (Filtered ≥5 voxels): {stats['num_lesions_filtered']}\n\n")

    text.insert(tk.END, f"Mean Intensity (Lesion, DWI): {stats['mean_intensity_filtered']:.4f}\n")
    text.insert(tk.END, f"Mean Intensity (Surrounding, DWI): {stats['mean_intensity_surrounding']:.4f}\n\n")

    text.insert(tk.END, f"Mean Intensity (Lesion, ADC): {stats['mean_intensity_adc']:.4f}\n")
    text.insert(tk.END, f"Mean Intensity (Surrounding, ADC): {stats['mean_intensity_surrounding_adc']:.4f}\n\n")

    text.insert(tk.END, "Filtered Lesion Volumes (cm³):\n")
    for v in stats["filtered_lesion_volumes"]:
        text.insert(tk.END, f"   - {v:.4f}\n")


def case_selector_gui(stats_all):
    """GUI to select one case and display its metrics."""
    root = tk.Tk()
    root.title("Case Selector")
    root.geometry("350x200")

    tk.Label(root, text="Select a case:", font=("Arial", 12)).pack(pady=10)

    case_ids = [s["case_id"] for s in stats_all]
    combo = ttk.Combobox(root, values=case_ids, state="readonly", font=("Arial", 12))
    combo.pack(pady=10)

    def show_selected():
        if not combo.get():
            messagebox.showwarning("Warning", "Select a case first.")
            return

        cid = combo.get()
        stats = next(s for s in stats_all if s["case_id"] == cid)
        show_case_details(stats)

    tk.Button(root, text="Show Case Details", command=show_selected,
              bg="#90caf9", font=("Arial", 12)).pack(pady=20)

    root.mainloop()


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():
    print("Select ISLES-2022 root folder...")
    root = tk.Tk()
    root.withdraw()
    root_dir = filedialog.askdirectory(title="Select ISLES-2022 Root Folder")
    if not root_dir:
        print("No folder selected.")
        return

    print("\nSearching for cases...")
    image_pairs = find_image_pairs(root_dir)
    print(f"Found {len(image_pairs)} cases.\n")

    stats_all = []

    for case_id, orig_path, mask_path, adc_path in image_pairs:
        print(f"Processing {case_id}...")

        orig = nib.load(orig_path)
        mask = nib.load(mask_path)
        adc = nib.load(adc_path) if adc_path else None

        stats = analyze_mask(mask, orig, adc)
        stats["case_id"] = case_id.replace("sub-", "")

        stats_all.append(stats)

    print("\nLaunching case selector GUI...")
    case_selector_gui(stats_all)


if __name__ == "__main__":
    main()
