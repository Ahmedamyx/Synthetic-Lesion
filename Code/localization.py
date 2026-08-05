#!/usr/bin/env python3
"""
Lesion Center Distribution Generator for ISLES Dataset

Analyzes lesion masks from ISLES dataset to create a spatial probability distribution
of where lesions are typically located based on their CENTER OF MASS.

Inputs:
- ISLES dataset root folder
- Derivatives folder (with masks)
- SynthSeg folder (with labelmaps)

Outputs:
- KDE .npz file with lesion CENTER location distribution
- Statistics about lesion center locations
"""

import os
import numpy as np
import nibabel as nib
import glob
import re
import datetime
from scipy.ndimage import zoom, label
from scipy.stats import gaussian_kde
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json

# ---------- User dialogs ----------

def ask_for_directory(title="Select Directory"):
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def ask_for_string(title, prompt, initial=""):
    root = tk.Tk(); root.withdraw()
    val = simpledialog.askstring(title, prompt, initialvalue=initial)
    root.destroy()
    return val

# ---------- File utilities ----------

def find_isles_cases(isles_root):
    """
    Find all valid cases in ISLES dataset structure.
    Returns list of dictionaries with case info.
    """
    cases = []
    
    # Look for case folders (sub-*)
    case_folders = glob.glob(os.path.join(isles_root, "sub-*"))
    
    for case_folder in case_folders:
        case_id = os.path.basename(case_folder)
        
        # Look for session folder
        ses_folder = os.path.join(case_folder, "ses-0001")
        if not os.path.exists(ses_folder):
            continue
            
        # Look for DWI file
        dwi_files = glob.glob(os.path.join(ses_folder, "dwi", "*_dwi.nii*")) + \
                    glob.glob(os.path.join(ses_folder, "dwi", "*_DWI.nii*"))
        
        if not dwi_files:
            continue
            
        # Look for ADC file
        adc_files = glob.glob(os.path.join(ses_folder, "dwi", "*_adc.nii*")) + \
                    glob.glob(os.path.join(ses_folder, "dwi", "*_ADC.nii*"))
        
        dwi_path = dwi_files[0]
        adc_path = adc_files[0] if adc_files else None
        
        cases.append({
            'case_id': case_id,
            'case_folder': case_folder,
            'dwi_path': dwi_path,
            'adc_path': adc_path
        })
    
    print(f"Found {len(cases)} cases in ISLES dataset")
    return cases

def find_masks_for_cases(cases, derivatives_folder):
    """
    Find mask files for each case in derivatives folder.
    """
    for case in cases:
        case_id = case['case_id']
        
        # Look for mask file
        mask_patterns = [
            os.path.join(derivatives_folder, case_id, "ses-0001", f"{case_id}_ses-0001_msk.nii*"),
            os.path.join(derivatives_folder, case_id, "ses-0001", f"{case_id}_ses-0001_mask.nii*"),
            os.path.join(derivatives_folder, case_id, "ses-0001", "*.nii*")
        ]
        
        mask_path = None
        for pattern in mask_patterns:
            mask_files = glob.glob(pattern)
            for mf in mask_files:
                if 'msk' in mf.lower() or 'mask' in mf.lower():
                    mask_path = mf
                    break
            if mask_path:
                break
        
        case['mask_path'] = mask_path
    
    # Count how many cases have masks
    cases_with_masks = [c for c in cases if c['mask_path']]
    print(f"Found masks for {len(cases_with_masks)}/{len(cases)} cases")
    
    return cases

def find_synthseg_for_cases(cases, synthseg_folder):
    """
    Find SynthSeg files for each case.
    """
    for case in cases:
        case_id = case['case_id']
        
        # Extract numeric ID from case_id
        numeric_id = None
        match = re.search(r'strokecase(\d+)', case_id)
        if match:
            numeric_id = match.group(1)
        else:
            match = re.search(r'sub-(\d+)', case_id)
            if match:
                numeric_id = match.group(1)
            else:
                numbers = re.findall(r'\d+', case_id)
                if numbers:
                    numeric_id = numbers[-1]
        
        # Build search patterns
        search_patterns = []
        
        # Exact case ID match in folder structure
        search_patterns.append(os.path.join(synthseg_folder, case_id, f"*{case_id}*synthseg*.nii*"))
        search_patterns.append(os.path.join(synthseg_folder, case_id, f"*{case_id}*.nii*"))
        
        # If we have a numeric ID, try patterns with that
        if numeric_id:
            search_patterns.append(os.path.join(synthseg_folder, f"*{numeric_id}*", f"*{numeric_id}*synthseg*.nii*"))
            search_patterns.append(os.path.join(synthseg_folder, f"*{numeric_id}*", f"*{numeric_id}*.nii*"))
            
            for folder_pattern in [f"*strokecase{numeric_id}*", f"*sub-*{numeric_id}*", f"*{numeric_id}*"]:
                search_patterns.append(os.path.join(synthseg_folder, folder_pattern, f"*synthseg*.nii*"))
                search_patterns.append(os.path.join(synthseg_folder, folder_pattern, f"*{numeric_id}*synthseg*.nii*"))
        
        # Generic search in all subdirectories
        search_patterns.append(os.path.join(synthseg_folder, "*", f"*{case_id}*synthseg*.nii*"))
        search_patterns.append(os.path.join(synthseg_folder, "*", f"*synthseg*.nii*"))
        
        synthseg_path = None
        searched_paths = set()
        
        for pattern in search_patterns:
            synthseg_files = glob.glob(pattern)
            for sf in synthseg_files:
                if sf in searched_paths:
                    continue
                searched_paths.add(sf)
                
                filename = os.path.basename(sf).lower()
                dirname = os.path.basename(os.path.dirname(sf)).lower()
                
                is_synthseg = ('synthseg' in filename or 'seg' in filename or 
                              'label' in filename or 'parc' in filename)
                
                matches_case = (case_id.lower() in filename or 
                               case_id.lower() in dirname or
                               (numeric_id and numeric_id in filename))
                
                if is_synthseg and matches_case:
                    synthseg_path = sf
                    break
            
            if synthseg_path:
                break
        
        # If still not found, try a more aggressive search
        if not synthseg_path:
            all_nii_files = glob.glob(os.path.join(synthseg_folder, "**", "*.nii*"), recursive=True)
            
            for sf in all_nii_files:
                filename = os.path.basename(sf).lower()
                
                if 'synthseg' in filename:
                    path_lower = sf.lower()
                    if (case_id.lower() in path_lower or 
                        (numeric_id and numeric_id in path_lower)):
                        synthseg_path = sf
                        break
        
        case['synthseg_path'] = synthseg_path
        
        if synthseg_path:
            print(f"  Found SynthSeg for {case_id}: {os.path.basename(synthseg_path)}")
        else:
            print(f"  WARNING: No SynthSeg found for {case_id}")
    
    cases_with_synthseg = [c for c in cases if c['synthseg_path']]
    print(f"\nFound SynthSeg for {len(cases_with_synthseg)}/{len(cases)} cases")
    
    return cases

# ---------- Analysis functions ----------

def analyze_lesion_centers(case):
    """
    Analyze lesion CENTER locations for a single case.
    For each lesion (connected component), calculates its center of mass
    and records the normalized coordinates and SynthSeg region.
    
    Returns:
    - lesion_centers: List of normalized center coordinates
    - lesion_stats_list: List of statistics for each lesion
    """
    try:
        # Load DWI for reference dimensions
        dwi_img = nib.load(case['dwi_path'])
        dwi_data = dwi_img.get_fdata()
        if dwi_data.ndim == 4:
            dwi_data = dwi_data[..., 0]
        
        # Load mask
        if not case['mask_path']:
            return None, None
        
        mask_img = nib.load(case['mask_path'])
        mask_data = mask_img.get_fdata()
        
        if mask_data.ndim == 4:
            mask_data = mask_data[..., 0]
        
        # Binarize mask
        lesion_mask = mask_data > 0
        
        if np.sum(lesion_mask) == 0:
            print(f"  No lesions found in {case['case_id']}")
            return None, None
        
        # Identify individual lesions using connected components
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled_mask, num_lesions = label(lesion_mask, structure=structure)
        
        print(f"  {case['case_id']}: Found {num_lesions} lesions")
        
        # Load SynthSeg for anatomical reference
        if not case['synthseg_path']:
            return None, None
        
        synthseg_img = nib.load(case['synthseg_path'])
        label_data = synthseg_img.get_fdata().astype(np.int32)
        
        # Resample labelmap to match DWI if needed
        if label_data.shape != dwi_data.shape:
            zoom_factors = np.array(dwi_data.shape) / np.array(label_data.shape)
            label_data = zoom(label_data, zoom_factors, order=0)
        
        # Store CENTER coordinates for all lesions
        lesion_centers = []
        lesion_stats_list = []
        
        # Analyze each lesion separately
        for lesion_idx in range(1, num_lesions + 1):
            # Create mask for this specific lesion
            single_lesion_mask = (labeled_mask == lesion_idx)
            
            # Get lesion voxel coordinates
            lesion_coords = np.argwhere(single_lesion_mask)
            
            if len(lesion_coords) == 0:
                continue
            
            # Calculate CENTER OF MASS (mean of coordinates)
            center_coords = np.mean(lesion_coords, axis=0)
            
            # Get SynthSeg label at center coordinate
            center_x, center_y, center_z = int(center_coords[0]), int(center_coords[1]), int(center_coords[2])
            
            # Ensure coordinates are within bounds
            center_x = max(0, min(center_x, label_data.shape[0] - 1))
            center_y = max(0, min(center_y, label_data.shape[1] - 1))
            center_z = max(0, min(center_z, label_data.shape[2] - 1))
            
            center_label = label_data[center_x, center_y, center_z]
            
            # Normalize center coordinates to [0, 1] range for KDE
            normalized_center = center_coords.astype(np.float32)
            for i in range(3):
                if dwi_data.shape[i] > 1:
                    normalized_center[i] = normalized_center[i] / (dwi_data.shape[i] - 1)
            
            # Get voxel spacing for volume calculation
            voxel_spacing = dwi_img.header.get_zooms()[:3]
            voxel_volume_mm3 = np.prod(voxel_spacing)
            lesion_volume_mm3 = len(lesion_coords) * voxel_volume_mm3
            
            # Store center coordinates
            lesion_centers.append(normalized_center)
            
            # Store lesion statistics
            lesion_stats_list.append({
                'lesion_index': lesion_idx,
                'case_id': case['case_id'],
                'voxel_count': len(lesion_coords),
                'volume_mm3': lesion_volume_mm3,
                'center_normalized': normalized_center.tolist(),
                'center_voxel': [center_x, center_y, center_z],
                'synthseg_label': int(center_label),
                'center_label_name': f"Label_{center_label}"
            })
            
            print(f"    Lesion {lesion_idx}: Center at [{center_x}, {center_y}, {center_z}] in label {center_label}, {len(lesion_coords)} voxels ({lesion_volume_mm3:.1f} mm³)")
        
        if lesion_centers:
            case_centers = np.vstack(lesion_centers)
            return case_centers, lesion_stats_list
        else:
            return None, None
        
    except Exception as e:
        print(f"  Error processing {case['case_id']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_center_kde(all_centers, bandwidth=0.1):
    """
    Create KDE from all lesion CENTER coordinates.
    all_centers is a list of numpy arrays, each containing center coordinates for one case.
    """
    if len(all_centers) == 0:
        return None
    
    # Flatten all center coordinates for KDE
    centers_list = []
    for centers in all_centers:
        if centers is not None and len(centers) > 0:
            centers_list.append(centers)
    
    if len(centers_list) == 0:
        return None
    
    # Combine all center coordinates
    centers_array = np.vstack(centers_list).T  # Shape: (3, num_lesions)
    
    # Create KDE
    kde = gaussian_kde(centers_array, bw_method=bandwidth)
    
    return kde

def save_kde_npz(kde, output_path, metadata):
    """
    Save KDE to NPZ file with metadata.
    """
    # Get the dataset (center coordinates)
    dataset = kde.dataset
    
    np.savez(output_path,
             dataset=dataset,
             bw_method=kde.factor,
             metadata=json.dumps(metadata))
    
    print(f"Saved KDE to: {output_path}")

def save_center_statistics(lesion_stats_all, output_dir):
    """
    Save detailed statistics about lesion centers.
    """
    stats_dir = os.path.join(output_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    if lesion_stats_all:
        # Calculate summary statistics
        total_lesions = len(lesion_stats_all)
        total_cases = len(set([ld['case_id'] for ld in lesion_stats_all]))
        
        # Count lesions per region
        region_counts = {}
        for lesion_data in lesion_stats_all:
            label_val = lesion_data['synthseg_label']
            if label_val not in region_counts:
                region_counts[label_val] = 0
            region_counts[label_val] += 1
        
        # Sort regions by count
        sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate center coordinate statistics
        all_centers = np.array([ld['center_normalized'] for ld in lesion_stats_all])
        center_means = np.mean(all_centers, axis=0).tolist()
        center_stds = np.std(all_centers, axis=0).tolist()
        
        summary_stats = {
            'total_cases': total_cases,
            'total_lesions': total_lesions,
            'center_coordinate_statistics': {
                'mean_normalized': center_means,
                'std_normalized': center_stds
            },
            'region_distribution': [
                {
                    'label': int(label_val),
                    'count': int(count),
                    'percentage': float(count / total_lesions * 100)
                }
                for label_val, count in sorted_regions
            ],
            'volume_statistics': {
                'mean_mm3': float(np.mean([ld['volume_mm3'] for ld in lesion_stats_all])),
                'median_mm3': float(np.median([ld['volume_mm3'] for ld in lesion_stats_all])),
                'min_mm3': float(np.min([ld['volume_mm3'] for ld in lesion_stats_all])),
                'max_mm3': float(np.max([ld['volume_mm3'] for ld in lesion_stats_all])),
                'std_mm3': float(np.std([ld['volume_mm3'] for ld in lesion_stats_all]))
            }
        }
        
        # Save detailed lesion data
        detailed_data = []
        for lesion_data in lesion_stats_all:
            detailed_data.append({
                'case_id': lesion_data['case_id'],
                'lesion_index': lesion_data['lesion_index'],
                'voxel_count': lesion_data['voxel_count'],
                'volume_mm3': lesion_data['volume_mm3'],
                'center_normalized': lesion_data['center_normalized'],
                'center_voxel': lesion_data['center_voxel'],
                'synthseg_label': lesion_data['synthseg_label']
            })
        
        # Save summary statistics
        summary_path = os.path.join(stats_dir, "summary_statistics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary statistics to: {summary_path}")
        
        # Save detailed lesion data
        detailed_path = os.path.join(stats_dir, "detailed_lesion_data.json")
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        print(f"Saved detailed lesion data to: {detailed_path}")
        
        # Print top regions
        print(f"\nTop 10 regions by lesion count:")
        for i, (label_val, count) in enumerate(sorted_regions[:10]):
            percentage = count / total_lesions * 100
            print(f"  {i+1}. Label {label_val}: {count} lesions ({percentage:.1f}%)")

# ---------- Main function ----------

def main():
    print("\n" + "="*80)
    print("ISLES Dataset Lesion CENTER Distribution Generator")
    print("="*80)
    print("Creates KDE distribution based on CENTER OF MASS of each lesion")
    print("Registered to SynthSeg anatomical labels")
    print("="*80)
    
    print("\nSelect ISLES dataset root folder:")
    isles_root = ask_for_directory("Select ISLES dataset root folder")
    if not isles_root:
        print("No folder selected. Exiting.")
        return
    
    print("\nSelect derivatives folder (containing masks):")
    derivatives_folder = ask_for_directory("Select derivatives folder")
    if not derivatives_folder:
        print("No folder selected. Exiting.")
        return
    
    print("\nSelect SynthSeg folder (containing labelmaps):")
    synthseg_folder = ask_for_directory("Select SynthSeg folder")
    if not synthseg_folder:
        print("No folder selected. Exiting.")
        return
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(isles_root, f"lesion_center_distribution_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nScanning ISLES dataset at: {isles_root}")
    cases = find_isles_cases(isles_root)
    
    if not cases:
        print("No cases found in ISLES dataset. Exiting.")
        return
    
    cases = find_masks_for_cases(cases, derivatives_folder)
    cases = find_synthseg_for_cases(cases, synthseg_folder)
    
    valid_cases = []
    for case in cases:
        if case['mask_path'] and case['synthseg_path']:
            valid_cases.append(case)
    
    print(f"\nFound {len(valid_cases)} cases with masks and SynthSeg files")
    
    if len(valid_cases) == 0:
        print("No valid cases found. Exiting.")
        return
    
    process_all = ask_for_string("Process cases", "Process all cases? (y/n)", "y")
    
    if process_all.lower() != 'y':
        print("\nAvailable cases:")
        for i, case in enumerate(valid_cases):
            print(f"{i+1}. {case['case_id']}")
        
        selection = ask_for_string("Select cases", 
                                  f"Enter case numbers (comma-separated, 1-{len(valid_cases)}) or 'all':", 
                                  "all")
        
        if selection.lower() == 'all':
            selected_cases = valid_cases
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                selected_cases = [valid_cases[i] for i in indices if 0 <= i < len(valid_cases)]
            except:
                print("Invalid selection. Processing all cases.")
                selected_cases = valid_cases
    else:
        selected_cases = valid_cases
    
    print(f"\nProcessing {len(selected_cases)} cases...")
    
    all_centers = []
    all_lesion_stats = []
    
    for i, case in enumerate(selected_cases):
        print(f"\nAnalyzing case {i+1}/{len(selected_cases)}: {case['case_id']}")
        
        centers, lesion_stats_list = analyze_lesion_centers(case)
        
        if centers is not None and len(centers) > 0:
            all_centers.append(centers)
            if lesion_stats_list:
                all_lesion_stats.extend(lesion_stats_list)
    
    print(f"\nAnalysis complete.")
    print(f"Processed {len(all_centers)} cases with lesions")
    print(f"Total individual lesions analyzed: {len(all_lesion_stats)}")
    
    if len(all_centers) == 0:
        print("No lesion data found. Exiting.")
        return
    
    # Calculate total lesion centers
    total_lesion_centers = sum(len(centers) for centers in all_centers)
    
    print(f"\nCreating spatial KDE from {len(all_centers)} cases...")
    print(f"Total lesion centers: {total_lesion_centers}")
    
    bandwidth = ask_for_string("KDE bandwidth", 
                              "Enter KDE bandwidth (default: 0.1, lower=sharper, higher=smoother):", 
                              "0.1")
    
    try:
        bandwidth = float(bandwidth)
    except:
        bandwidth = 0.1
        print(f"Using default bandwidth: {bandwidth}")
    
    kde = create_center_kde(all_centers, bandwidth=bandwidth)
    
    if kde is None:
        print("Failed to create KDE. Exiting.")
        return
    
    # Get reference shape from first case
    ref_img = nib.load(selected_cases[0]['dwi_path'])
    ref_shape = ref_img.shape[:3]
    
    metadata = {
        'dataset': 'ISLES',
        'analysis_type': 'lesion_center_distribution',
        'analysis_date': timestamp,
        'num_cases': len(selected_cases),
        'num_cases_with_lesions': len(all_centers),
        'total_individual_lesions': len(all_lesion_stats),
        'total_lesion_centers': int(total_lesion_centers),
        'bandwidth': bandwidth,
        'reference_shape': ref_shape,
        'coordinate_space': 'normalized_0_1',
        'center_definition': 'center_of_mass',
        'cases_processed': [case['case_id'] for case in selected_cases]
    }
    
    kde_path = os.path.join(output_dir, "lesion_center_distribution.npz")
    save_kde_npz(kde, kde_path, metadata)
    
    print("\nSaving statistics...")
    save_center_statistics(all_lesion_stats, output_dir)
    
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        summary = {
            'metadata': metadata,
            'kde_file': os.path.basename(kde_path),
            'bandwidth_used': bandwidth
        }
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"KDE file: {kde_path}")
    print(f"Number of cases analyzed: {len(all_centers)}")
    print(f"Total lesion centers: {metadata['total_lesion_centers']}")
    print(f"Bandwidth used: {bandwidth}")
    print("="*80)
    
    messagebox.showinfo(
        "Analysis Complete",
        f"Lesion CENTER distribution analysis complete!\n\n"
        f"Analyzed {len(all_centers)} cases\n"
        f"Total lesion centers: {metadata['total_lesion_centers']}\n\n"
        f"Outputs saved to:\n{output_dir}"
    )

if __name__ == "__main__":
    main()