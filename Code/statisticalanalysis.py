import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_dilation
from nilearn import plotting
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde
import inspect

def save_kde_model(values, filename):
    """
    Save KDE model to a .npz file (safer than pickle).
    Stores dataset and bw.
    """
    kde = gaussian_kde(values)
    np.savez(filename, dataset=kde.dataset, bw_method=kde.covariance_factor())
    return kde

def compute_kde(values, bw_scale=2.0, method='scott'):
    kde = gaussian_kde(values, bw_method=method)
    kde.set_bandwidth(kde.factor * bw_scale)

    grid = np.linspace(min(values), max(values), 500)
    kde_vals = kde(grid)

    return kde, grid, kde_vals


def analyze_mask(mask_img, original_img, adc_img, min_voxels=5, dilation_size=5):
    mask_data = mask_img.get_fdata()
    orig_data = original_img.get_fdata()

    if adc_img is not None:
        adc_data_raw = adc_img.get_fdata()
        if adc_data_raw.max() > 10:
            adc_data = adc_data_raw / 1000.0
        else:
            adc_data = adc_data_raw
    else:
        adc_data = None

    lesion_mask = (mask_data > 0)
    voxel_volume_mm3 = np.prod(mask_img.header.get_zooms())
    num_voxels = np.sum(lesion_mask)
    volume_mm3 = num_voxels * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, num_lesions = label(lesion_mask, structure=structure)

    lesion_sizes_voxels = np.bincount(labeled_mask.flatten())[1:]  # Exclude background

    # Filter lesions smaller than min_voxels
    filtered_labels = [i + 1 for i, size in enumerate(lesion_sizes_voxels) if size >= min_voxels]
    filtered_mask = np.isin(labeled_mask, filtered_labels)

    # Calculate individual lesion volumes in mm3 for filtered lesions
    filtered_lesion_volumes = []
    for label_idx in filtered_labels:
        size_vox = lesion_sizes_voxels[label_idx - 1]
        vol_mm3 = size_vox * voxel_volume_mm3
        filtered_lesion_volumes.append(vol_mm3 / 1000.0)  # cm3

    mean_intensity_filtered = orig_data[filtered_mask].mean() if np.sum(filtered_mask) > 0 else 0.0
    mean_intensity_adc = adc_data[filtered_mask].mean() if adc_data is not None and np.sum(filtered_mask) > 0 else 0.0

    # Surrounding tissue calculations (unchanged)
    if np.sum(filtered_mask) == 0:
        mean_intensity_surrounding = 0.0
        mean_intensity_surrounding_adc = 0.0
    else:
        struct_elem = np.ones((dilation_size, dilation_size, dilation_size), dtype=np.uint8)
        dilated_mask = binary_dilation(filtered_mask, structure=struct_elem)
        surrounding_mask = dilated_mask & (~filtered_mask)
        if np.sum(surrounding_mask) > 0:
            mean_intensity_surrounding = orig_data[surrounding_mask].mean()
            mean_intensity_surrounding_adc = adc_data[surrounding_mask].mean() if adc_data is not None else 0.0
        else:
            mean_intensity_surrounding = 0.0
            mean_intensity_surrounding_adc = 0.0

    return {
        'volume_mm3': volume_mm3,
        'volume_cm3': volume_cm3,
        'num_voxels': int(num_voxels),
        'num_lesions_total': int(num_lesions),
        'num_lesions_filtered': len(filtered_labels),
        'mean_intensity_filtered': float(mean_intensity_filtered),
        'mean_intensity_adc': float(mean_intensity_adc),
        'mean_intensity_surrounding': float(mean_intensity_surrounding),
        'mean_intensity_surrounding_adc': float(mean_intensity_surrounding_adc),
        'filtered_mask': filtered_mask,
        'filtered_lesion_volumes': filtered_lesion_volumes  # NEW
    }


def find_image_pairs(root_dir):
    image_pairs = []
    derivatives_dir = os.path.join(root_dir, "derivatives")

    for case in os.listdir(derivatives_dir):
        case_dir = os.path.join(derivatives_dir, case, "ses-0001")
        if not os.path.isdir(case_dir):
            continue
        mask_file = os.path.join(case_dir, f"{case}_ses-0001_msk.nii.gz")
        if not os.path.exists(mask_file):
            continue

        original_file = os.path.join(root_dir, case, "ses-0001", "dwi", f"{case}_ses-0001_dwi.nii.gz")
        if not os.path.exists(original_file):
            continue

        adc_file = os.path.join(root_dir, case, "ses-0001", "dwi", f"{case}_ses-0001_adc.nii.gz")
        if not os.path.exists(adc_file):
            adc_file = None

        image_pairs.append((original_file, mask_file, adc_file))

    return image_pairs

def detect_outliers_tukey(data, multiplier=3.0):
    """Detect outliers using Tukey's rule (IQR method)"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outliers = [(i, val) for i, val in enumerate(data) if val < lower_bound or val > upper_bound]
    return outliers, lower_bound, upper_bound

def plot_hist(stat_list, title, xlabel, outlier_indices=None, case_ids=None):
    """
    Histogram with KDE overlay + saved KDE model.
    """
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(stat_list, bins=30, color='skyblue', edgecolor='black', density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True)
    plt.tight_layout()

    # === KDE computation + plot ===
    kde, kde_x, kde_y = compute_kde(stat_list)
    ax.plot(kde_x, kde_y, linewidth=2)

    # === Save KDE to disk ===
    kde_filename = f"{title.replace(' ', '_').lower()}_kde.npz"
    kde_filename = os.path.join(os.getcwd(), kde_filename)
    save_kde_model(stat_list, kde_filename)
    print(f"[KDE saved] → {kde_filename}")

    # Hover remains unchanged
    bin_to_cases = {i: [] for i in range(len(n))}
    if case_ids is None:
        case_ids = [f"Case {i+1}" for i in range(len(stat_list))]

    for val, cid in zip(stat_list, case_ids):
        for i in range(len(bins) - 1):
            if (bins[i] <= val < bins[i + 1]) or (i == len(bins) - 2 and val == bins[-1]):
                bin_to_cases[i].append(cid)
                break

    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(rect, i):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height()
        annot.xy = (x, y)
        cases_in_bin = bin_to_cases.get(i, [])
        text = f"{len(cases_in_bin)} case(s)\n" + ", ".join(cases_in_bin[:8])
        if len(cases_in_bin) > 8:
            text += ", ..."
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor("lightyellow")

    def hover(event):
        visible = annot.get_visible()
        if event.inaxes == ax:
            for i, rect in enumerate(patches):
                contains, _ = rect.contains(event)
                if contains:
                    update_annot(rect, i)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)




def plot_dual_histogram(data1, data2, label1, label2, title, xlabel):
    plt.figure()
    plt.hist(data1, bins=50, alpha=0.5, color='blue', edgecolor='black', label=label1)
    plt.hist(data2, bins=50, alpha=0.5, color='orange', edgecolor='black', label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def print_summary_stats(data, label):
    data = np.array(data)
    print(f"\n Statistics for {label}:")
    print(f" - Mean:       {np.mean(data):.4f}")
    print(f" - Median:     {np.median(data):.4f}")
    print(f" - Std Dev:    {np.std(data):.4f}")
    print(f" - Variance:   {np.var(data):.4f}")
    print(f" - Min:        {np.min(data):.4f}")
    print(f" - Max:        {np.max(data):.4f}")
    print(f" - IQR:        {np.percentile(data, 75) - np.percentile(data, 25):.4f}")
    print(f" - 10th %ile:  {np.percentile(data, 10):.4f}")
    print(f" - 25th %ile:  {np.percentile(data, 25):.4f}")
    print(f" - 75th %ile:  {np.percentile(data, 75):.4f}")
    print(f" - 90th %ile:  {np.percentile(data, 90):.4f}")
    print(f" - 95th %ile:  {np.percentile(data, 95):.4f}")
def enable_hover_labels(fig, ax, data_values, case_labels=None):
    """
    Ajoute une interaction de survol pour afficher les indices/cas associés à un histogramme.
    - data_values: liste des valeurs utilisées dans l'histogramme
    - case_labels: liste optionnelle des noms ou numéros de cas (sinon, indices)
    """
    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(data_values))]

    # Créer l'annotation (texte flottant)
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # Récupérer les patches de l'histogramme
    bars = [rect for rect in ax.patches]

    def update_annot(rect, label):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        annot.xy = (x, y)
        text = f"{label}\nValue = {rect.get_height():.2f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor("lightyellow")

    def hover(event):
        visible = annot.get_visible()
        if event.inaxes == ax:
            for i, rect in enumerate(bars):
                contains, _ = rect.contains(event)
                if contains:
                    update_annot(rect, case_labels[i])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
def enable_hover_heatmap(fig, ax, matrix, x_labels, y_labels, contributors):
    """
    Adds interactive hover for heatmap cells, showing which cases contribute to each probability.
    """
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    def hover(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return

            j = int((x - x_min) / ((x_max - x_min) / matrix.shape[1]))
            i = int((y - y_min) / ((y_max - y_min) / matrix.shape[0]))

            if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
                val = matrix[i, j]
                case_list = contributors[i][j]
                case_text = ", ".join(case_list[:8]) + (" ..." if len(case_list) > 8 else "")
                annot.xy = (x, y)
                annot.set_text(f"Lesions={x_labels[j]}\nVol>={y_labels[i]:.2f} cm³\nP={val:.3f}\n{len(case_list)} case(s): {case_text}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def plot_max_lesion_volume_probability_heatmap(stats_all):
    # Organize lesion volumes by lesion count Y
    lesion_volumes_by_count = {}
    for s in stats_all:
        Y = s['num_lesions_filtered']
        vols = s.get('filtered_lesion_volumes', [])
        cid = s.get('case_id', f"Case?")
        if Y not in lesion_volumes_by_count:
            lesion_volumes_by_count[Y] = []
        lesion_volumes_by_count[Y].append((cid, vols))

    lesion_counts = sorted(lesion_volumes_by_count.keys())

    # Get global min/max lesion volume across all lesions for thresholds
    all_lesion_volumes = [vol for vols_list in lesion_volumes_by_count.values()
                          for _, case_vols in vols_list for vol in case_vols]
    if not all_lesion_volumes:
        print("No lesion volumes found for heatmap.")
        return

    min_vol = min(all_lesion_volumes)
    max_vol = max(all_lesion_volumes)
    volume_thresholds = np.linspace(min_vol, max_vol, 50)

    prob_matrix = np.zeros((len(volume_thresholds), len(lesion_counts)))
    contributors = [[[] for _ in lesion_counts] for _ in volume_thresholds]  # track case IDs

    for j, Y in enumerate(lesion_counts):
        cases_vols = lesion_volumes_by_count[Y]
        for i, X in enumerate(volume_thresholds):
            count_cases_with_large_lesion = 0
            contributing_cases = []
            for cid, vols in cases_vols:
                if any(vol > X for vol in vols):
                    count_cases_with_large_lesion += 1
                    contributing_cases.append(cid)
            total_cases = len(cases_vols)
            prob_matrix[i, j] = count_cases_with_large_lesion / total_cases if total_cases > 0 else 0.0
            contributors[i][j] = contributing_cases

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(prob_matrix, aspect='auto', origin='lower',
                   extent=[min(lesion_counts)-0.5, max(lesion_counts)+0.5, min_vol, max_vol],
                   cmap='plasma')

    plt.colorbar(im, label='P(max lesion volume > X | lesions = Y)')
    plt.xlabel('Filtered Lesion Count (Y)')
    plt.ylabel('Lesion Volume Threshold (cm³) (X)')
    plt.title('Probability of At Least One Lesion > Volume Threshold\nGiven Number of Lesions')
    plt.xticks(lesion_counts)
    plt.tight_layout()

    enable_hover_heatmap(fig, ax, prob_matrix, lesion_counts, volume_thresholds, contributors)

def plot_conditional_probability_heatmap(num_lesions_filtered_list, volume_cm3_list, case_ids):
    lesion_counts = sorted(set(num_lesions_filtered_list))
    min_vol = min(volume_cm3_list)
    max_vol = max(volume_cm3_list)
    volume_thresholds = np.linspace(min_vol, max_vol, 50)

    prob_matrix = np.zeros((len(volume_thresholds), len(lesion_counts)))
    contributors = [[[] for _ in lesion_counts] for _ in volume_thresholds]

    for j, Y in enumerate(lesion_counts):
        indices = [i for i, val in enumerate(num_lesions_filtered_list) if val == Y]
        for i, X in enumerate(volume_thresholds):
            vols_for_Y = [volume_cm3_list[k] for k in indices]
            cases_for_Y = [case_ids[k] for k in indices]
            if len(vols_for_Y) == 0:
                prob_matrix[i, j] = 0.0
            else:
                hits = [cases_for_Y[k] for k, v in enumerate(vols_for_Y) if v > X]
                prob_matrix[i, j] = len(hits) / len(vols_for_Y)
                contributors[i][j] = hits

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(prob_matrix, aspect='auto', origin='lower',
                   extent=[min(lesion_counts)-0.5, max(lesion_counts)+0.5, min_vol, max_vol],
                   cmap='viridis')

    plt.colorbar(im, label='P(volume > X | lesions = Y)')
    plt.xlabel('Filtered Lesion Count (Y)')
    plt.ylabel('Volume Threshold (cm³) (X)')
    plt.title('Probability of total volume given number of lesions')
    plt.xticks(lesion_counts)
    plt.tight_layout()

    enable_hover_heatmap(fig, ax, prob_matrix, lesion_counts, volume_thresholds, contributors)

def save_outlier_report(filename, outliers_vol, vol_lower, vol_upper,
                        outliers_lesion, lesion_lower, lesion_upper,
                        combined_outlier_indices,
                        volume_cm3_list, num_lesions_filtered_list):
    with open(filename, 'w') as f:
        f.write("Detailed Outlier Report\n")
        f.write("="*30 + "\n\n")

        f.write("Cases flagged as Volume Outliers:\n")
        for idx, val in outliers_vol:
            reason = []
            if val < vol_lower:
                reason.append(f"volume below lower bound ({val:.4f} < {vol_lower:.4f})")
            elif val > vol_upper:
                reason.append(f"volume above upper bound ({val:.4f} > {vol_upper:.4f})")
            f.write(f"  Case {idx + 1}: Volume = {val:.4f} cm³, Filtered Lesions = {num_lesions_filtered_list[idx]} --> {' and '.join(reason)}\n")

        f.write("\nCases flagged as Lesion Count Outliers:\n")
        for idx, val in outliers_lesion:
            reason = []
            if val < lesion_lower:
                reason.append(f"lesion count below lower bound ({val} < {lesion_lower:.4f})")
            elif val > lesion_upper:
                reason.append(f"lesion count above upper bound ({val} > {lesion_upper:.4f})")
            f.write(f"  Case {idx + 1}: Lesion Count = {val}, Volume = {volume_cm3_list[idx]:.4f} cm³ --> {' and '.join(reason)}\n")

        f.write("\nCases flagged as Combined Outliers (both volume and lesion count):\n")
        for idx in combined_outlier_indices:
            vol_val = volume_cm3_list[idx]
            lesion_val = num_lesions_filtered_list[idx]
            f.write(f"  Case {idx + 1}: Volume = {vol_val:.4f} cm³, Filtered Lesions = {lesion_val} (both above thresholds)\n")
def choose_mode_gui():
    """Show a simple Tkinter GUI to choose analysis mode and return the selection."""
    import threading

    mode_selection = {'mode': None}

    def select_mode(selected_mode):
        mode_selection['mode'] = selected_mode
        root.quit()  # Exit the mainloop

    root = tk.Tk()
    root.title("Choose Analysis Mode")
    root.geometry("300x150")
    root.eval('tk::PlaceWindow . center')

    label = tk.Label(root, text="Select analysis mode:", font=("Arial", 12))
    label.pack(pady=10)

    btn_normal = tk.Button(root, text="Normal (no outliers)", width=20, command=lambda: select_mode('normal'), bg="#8ecae6")
    btn_normal.pack(pady=5)

    btn_outlier = tk.Button(root, text="Outliers only", width=20, command=lambda: select_mode('outlier'), bg="#f28482")
    btn_outlier.pack(pady=5)

    # Run the Tkinter main loop (blocking), but allow exit on button click
    root.mainloop()

    root.destroy()
    return mode_selection['mode']


def main():
    root = tk.Tk()
    root.withdraw()

    # Choose mode at start
    mode = choose_mode_gui()
    if mode not in ['normal', 'outlier']:
        print("No mode selected. Exiting.")
        return

    if mode not in ['normal', 'outlier']:
        print("Invalid mode selected. Please enter 'normal' or 'outlier'. Exiting.")
        return

    root_dir = filedialog.askdirectory(title="Select the ISLES-2022 root folder")
    if not root_dir:
        print("No directory selected.")
        return

    print(f"\nSearching for image-mask pairs in: {root_dir}")
    image_pairs = find_image_pairs(root_dir)
    print(f"Found {len(image_pairs)} valid image-mask pairs.\n")

    stats_all = []

    for idx, (orig_path, mask_path, adc_path) in enumerate(image_pairs):
        print(f"Analyzing case {idx + 1}/{len(image_pairs)}: {os.path.basename(orig_path)}")
        try:
            original_img = nib.load(orig_path)
            mask_img = nib.load(mask_path)
            adc_img = nib.load(adc_path) if adc_path else None
            
            stats = analyze_mask(mask_img, original_img, adc_img)
            stats['case_id'] = os.path.basename(orig_path).split("_")[0].replace("sub-", "")


            # Skip if volume or mean intensity is 0
            if stats['volume_mm3'] > 0 and stats['mean_intensity_filtered'] > 0:
                # Check for unusually high ADC intensity in lesions
                if stats['mean_intensity_adc'] > 10:
                    print(f"⚠️  High ADC lesion intensity (>10): {os.path.basename(adc_path)} - {stats['mean_intensity_adc']:.4f}")

                stats_all.append(stats)
            else:
                print(f"Skipping case due to zero volume or intensity.")

        except Exception as e:
            print(f"Failed to process case {orig_path}: {e}")

    if not stats_all:
        print("No valid data processed.")
        return

    # Extract lists for stats
    volume_cm3_list = [s['volume_cm3'] for s in stats_all]
    num_lesions_filtered_list = [s['num_lesions_filtered'] for s in stats_all]

    # Detect outliers in lesion volume and lesion count (multiplier=3)
    outliers_vol, vol_lower, vol_upper = detect_outliers_tukey(volume_cm3_list, multiplier=3)
    outliers_lesion, lesion_lower, lesion_upper = detect_outliers_tukey(num_lesions_filtered_list, multiplier=3)

    outlier_indices_vol = set(idx for idx, _ in outliers_vol)
    outlier_indices_lesion = set(idx for idx, _ in outliers_lesion)

    # Union of outliers in either lesion volume OR lesion count
    combined_outlier_indices = sorted(outlier_indices_vol | outlier_indices_lesion)

    # Filter stats_all depending on user mode
    if mode == 'normal':
        # Keep only non-outliers
        filtered_indices = [i for i in range(len(stats_all)) if i not in combined_outlier_indices]
        print(f"\nRunning analysis on NORMAL cases (without outliers): {len(filtered_indices)} cases")
    else:
        # Keep only outliers
        filtered_indices = combined_outlier_indices
        print(f"\nRunning analysis on OUTLIER cases: {len(filtered_indices)} cases")

    if len(filtered_indices) == 0:
        print(f"No cases to analyze for mode '{mode}'. Exiting.")
        return

    # Filter stats lists accordingly
    stats_all_filtered = [stats_all[i] for i in filtered_indices]
    volume_cm3_filtered = [volume_cm3_list[i] for i in filtered_indices]
    num_lesions_filtered_filtered = [num_lesions_filtered_list[i] for i in filtered_indices]

    # Extract other lists from filtered data
    num_voxels_list = [s['num_voxels'] for s in stats_all_filtered]
    num_lesions_total_list = [s['num_lesions_total'] for s in stats_all_filtered]
    mean_intensity_filtered_list = [s['mean_intensity_filtered'] for s in stats_all_filtered]
    mean_intensity_adc_list = [s['mean_intensity_adc'] for s in stats_all_filtered if s['mean_intensity_adc'] > 0]
    mean_intensity_surrounding_list = [s['mean_intensity_surrounding'] for s in stats_all_filtered]
    mean_intensity_surrounding_adc_list = [s['mean_intensity_surrounding_adc'] for s in stats_all_filtered if s['mean_intensity_surrounding_adc'] > 0]

    # Proceed with analysis/plots/stats for filtered data only
    
    plot_max_lesion_volume_probability_heatmap(stats_all_filtered)
    case_ids = [s['case_id'] for s in stats_all_filtered]
    plot_conditional_probability_heatmap(num_lesions_filtered_filtered, volume_cm3_filtered, case_ids)

    case_ids = [s['case_id'] for s in stats_all_filtered]
    # Plot histograms
    plot_hist(num_voxels_list, "Distribution of Lesion Voxels", "Number of Voxels", case_ids=case_ids)
    plot_hist(num_lesions_total_list, "Distribution of Connected Lesions (Total)", "Lesion Count", case_ids=case_ids)
    plot_hist(num_lesions_filtered_filtered, "Distribution of Connected Lesions (≥ 5 voxels)", "Filtered Lesion Count", case_ids=case_ids)
    plot_hist(volume_cm3_filtered, "Distribution of Lesion Volume", "Volume (cm³)", case_ids=case_ids)

    # DWI intensity histograms
    plot_hist(mean_intensity_filtered_list, "DWI Intensity in Lesion", "Intensity", case_ids=case_ids)
    plot_hist(mean_intensity_surrounding_list, "DWI Intensity in Surrounding Tissue", "Intensity", case_ids=case_ids)

        # ADC intensity histograms — cleaned and merged
    adc_threshold = 0.04  # Filtre pour ignorer valeurs quasi nulles

    # Filtrer les ADC trop proches de zéro
    mean_intensity_adc_filtered_clean = [v for v in mean_intensity_adc_list if v > adc_threshold]
    mean_intensity_surrounding_adc_filtered_clean = [v for v in mean_intensity_surrounding_adc_list if v > adc_threshold]

    # Histogramme combiné
    # ADC histogram with KDE
    plt.figure()
    plt.hist(mean_intensity_adc_filtered_clean, bins=50, alpha=0.6, color='steelblue',
            edgecolor='black', density=True, label='Lesion ADC Intensity')

    plt.hist(mean_intensity_surrounding_adc_filtered_clean, bins=50, alpha=0.6, color='orange',
            edgecolor='black', density=True, label='Surrounding ADC Intensity')

    # KDE for lesion ADC
    if len(mean_intensity_adc_filtered_clean) > 1:
        kde_adc, x_adc, y_adc = compute_kde(mean_intensity_adc_filtered_clean)
        plt.plot(x_adc, y_adc, linewidth=2)
        save_kde_model(mean_intensity_adc_filtered_clean, os.path.join(os.getcwd(), "adc_lesion_kde.npz"))
        print("[KDE saved] → adc_lesion_kde.npz")

    # KDE for surrounding ADC
    if len(mean_intensity_surrounding_adc_filtered_clean) > 1:
        kde_sur, x_sur, y_sur = compute_kde(mean_intensity_surrounding_adc_filtered_clean)
        plt.plot(x_sur, y_sur, linewidth=2)
        save_kde_model(mean_intensity_surrounding_adc_filtered_clean, os.path.join(os.getcwd(), "adc_surrounding_kde.npz"))
        print("[KDE saved] → adc_surrounding_kde.npz")

    plt.title("ADC Intensity Distribution\n(Lesion vs Surrounding Tissue)")
    plt.xlabel("ADC Intensity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS ({mode.upper()} CASES)")
    print("="*60)
    print_summary_stats(volume_cm3_filtered, "Lesion Volume (cm³)")
    print_summary_stats(num_voxels_list, "Number of Lesion Voxels")
    print_summary_stats(num_lesions_total_list, "Number of Connected Lesions (Total)")
    print_summary_stats(num_lesions_filtered_filtered, "Number of Connected Lesions (Filtered ≥ 5 voxels)")
    print_summary_stats(mean_intensity_filtered_list, "Mean DWI Intensity in Filtered Lesions")
    print_summary_stats(mean_intensity_adc_list, "Mean ADC Intensity in Filtered Lesions")
    print_summary_stats(mean_intensity_surrounding_list, "Mean DWI Intensity in Surrounding Tissue")
    print_summary_stats(mean_intensity_surrounding_adc_list, "Mean ADC Intensity in Surrounding Tissue")

    plt.show()

if __name__ == "__main__":
    main()