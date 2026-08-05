import os
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from skimage.exposure import match_histograms


class HistogramMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Histogram Intensity Matching - ISLES to HCP")
        self.root.geometry("800x600")
        
        self.isles_path = None
        self.hcp_path = None
        self.output_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ISLES Dataset
        ttk.Label(main_frame, text="ISLES Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.isles_label = ttk.Label(main_frame, text="No folder selected", foreground="gray")
        self.isles_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_isles).grid(row=0, column=2, padx=5)
        
        # HCP Dataset
        ttk.Label(main_frame, text="HCP Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hcp_label = ttk.Label(main_frame, text="No folder selected", foreground="gray")
        self.hcp_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_hcp).grid(row=1, column=2, padx=5)
        
        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_label = ttk.Label(main_frame, text="No folder selected", foreground="gray")
        self.output_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to process", foreground="blue")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Process Histogram Matching", 
                                      command=self.process, state=tk.DISABLED)
        self.process_btn.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def browse_isles(self):
        folder = filedialog.askdirectory(title="Select ISLES Dataset Folder")
        if folder:
            self.isles_path = folder
            self.isles_label.config(text=os.path.basename(folder), foreground="black")
            self.check_ready()
    
    def browse_hcp(self):
        folder = filedialog.askdirectory(title="Select HCP Dataset Folder")
        if folder:
            self.hcp_path = folder
            self.hcp_label.config(text=os.path.basename(folder), foreground="black")
            self.check_ready()
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path = folder
            self.output_label.config(text=os.path.basename(folder), foreground="black")
            self.check_ready()
    
    def check_ready(self):
        if self.isles_path and self.hcp_path and self.output_path:
            self.process_btn.config(state=tk.NORMAL)
    
    def normalize_units_isles(self, data):
        """Normalize ISLES units to mm²/s based on median non-zero value with granular detection"""
        nonzero_data = data[data > 0]
        if len(nonzero_data) == 0:
            return data
        
        median_val = np.median(nonzero_data)
        print(f"    Median non-zero value: {median_val}")
        
        # Granular unit detection
        if 0.01 <= median_val <= 1:
            print(f"    -> Converting from 10^-3 m²/s to mm²/s (multiplying by 1000)")
            return data * 1000.0
        elif 0.0001 <= median_val <= 0.004:
            print(f"    -> Converting from 10^-6 m²/s to mm²/s (multiplying by 10^6)")
            return data * 1000000.0
        else:
            print(f"    -> Already in mm²/s or unrecognized range, no conversion")
            return data
    
    def load_isles_data(self):
        """Load all ISLES data (ADC and DWI) using sampling for efficiency"""
        adc_data = []
        dwi_data = []
        
        self.status_label.config(text="Loading ISLES data...")
        self.root.update()
        
        # Find all substroke folders
        try:
            substroke_folders = [f for f in Path(self.isles_path).iterdir() 
                               if f.is_dir() and 'sub-stroke' in f.name]
        except Exception as e:
            print(f"Error reading ISLES directory: {e}")
            substroke_folders = []
        
        print(f"Found {len(substroke_folders)} substroke folders")
        
        if len(substroke_folders) == 0:
            messagebox.showwarning("Warning", "No 'sub-stroke' folders found in ISLES directory!")
            return np.array([]), np.array([])
        
        file_count = 0
        max_samples_per_file = 100000  # Limit samples per file
        
        for folder in substroke_folders:
            # Look for ADC and DWI files
            nifti_files = list(folder.rglob('*.nii.gz'))
            print(f"Folder {folder.name}: {len(nifti_files)} .nii.gz files")
            
            for file in nifti_files:
                try:
                    # Skip if filename doesn't contain relevant keywords
                    filename_lower = file.name.lower()
                    if not any(keyword in filename_lower for keyword in ['adc', 'dwi', 'b1000']):
                        continue
                    
                    print(f"Loading: {file.name}")
                    self.status_label.config(text=f"Loading ISLES: {file.name}...")
                    self.root.update()
                    
                    img = nib.load(str(file))
                    data = img.get_fdata()
                    
                    # Normalize ISLES units based on median
                    max_val = np.max(data)
                    print(f"  Max value: {max_val}")
                    data = self.normalize_units_isles(data)
                    
                    # Sample data to avoid memory issues
                    data_flat = data.flatten()
                    
                    # Remove zeros first
                    data_nonzero = data_flat[data_flat > 0]
                    
                    # Subsample if too large
                    if len(data_nonzero) > max_samples_per_file:
                        step = len(data_nonzero) // max_samples_per_file
                        data_sample = data_nonzero[::step]
                    else:
                        data_sample = data_nonzero
                    
                    # Classify based on filename
                    if 'adc' in filename_lower:
                        adc_data.append(data_sample)
                        print(f"  -> ADC file, {len(data_sample)} samples")
                    elif 'dwi' in filename_lower or 'b1000' in filename_lower:
                        dwi_data.append(data_sample)
                        print(f"  -> DWI file, {len(data_sample)} samples")
                    
                    file_count += 1
                    
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Total files loaded: {file_count}")
        
        # Concatenate all data
        adc_combined = np.concatenate(adc_data) if adc_data else np.array([])
        dwi_combined = np.concatenate(dwi_data) if dwi_data else np.array([])
        
        print(f"ADC samples: {len(adc_combined)}, DWI samples: {len(dwi_combined)}")
        
        if len(adc_combined) == 0 and len(dwi_combined) == 0:
            messagebox.showwarning("Warning", 
                                 "No ADC or DWI files found in ISLES dataset!\n"
                                 "Make sure files contain 'adc', 'dwi', or 'b1000' in their names.")
        
        return adc_combined, dwi_combined
    
    def load_hcp_files(self):
        """Load all HCP files"""
        hcp_files = {'adc': [], 'dwi': []}
        
        for file in Path(self.hcp_path).glob('*_adc.nii.gz'):
            hcp_files['adc'].append(file)
        
        for file in Path(self.hcp_path).glob('*_b1000.nii.gz'):
            hcp_files['dwi'].append(file)
        
        return hcp_files
    
    def match_histogram_simple(self, source, reference):
        """Simple histogram matching using percentile mapping"""
        # Remove zeros
        source_nonzero = source[source > 0]
        reference_nonzero = reference[reference > 0]
        
        # Create output
        matched = np.zeros_like(source)
        mask = source > 0
        
        # Calculate percentiles
        percentiles = np.linspace(0, 100, 1000)
        source_percentiles = np.percentile(source_nonzero, percentiles)
        reference_percentiles = np.percentile(reference_nonzero, percentiles)
        
        # Map source values to reference distribution
        matched[mask] = np.interp(source[mask], source_percentiles, reference_percentiles)
        
        return matched
    
    def process(self):
        try:
            self.process_btn.config(state=tk.DISABLED)
            self.progress['value'] = 0
            
            # Load ISLES reference data
            self.status_label.config(text="Loading ISLES reference data...")
            self.root.update()
            isles_adc, isles_dwi = self.load_isles_data()
            self.progress['value'] = 20
            
            if len(isles_adc) == 0 and len(isles_dwi) == 0:
                messagebox.showerror("Error", "No ISLES data found!")
                return
            
            # Load HCP files
            self.status_label.config(text="Finding HCP files...")
            self.root.update()
            hcp_files = self.load_hcp_files()
            total_files = len(hcp_files['adc']) + len(hcp_files['dwi'])
            
            if total_files == 0:
                messagebox.showerror("Error", "No HCP files found!")
                return
            
            self.progress['value'] = 30
            
            # Create output directory
            os.makedirs(self.output_path, exist_ok=True)
            
            # Process HCP ADC files
            processed = 0
            matched_adc_data = []
            matched_dwi_data = []
            
            for adc_file in hcp_files['adc']:
                self.status_label.config(text=f"Processing {adc_file.name}...")
                self.root.update()
                
                img = nib.load(str(adc_file))
                data = img.get_fdata()
                # Convert HCP data from m²/s to mm²/s
                median_val = np.median(data[data > 0])
                print(f"Processing HCP ADC: {adc_file.name}, median before: {median_val}")
                data = data * 1000.0
                print(f"  -> Converted to mm²/s, median after: {np.median(data[data > 0])}")
                
                if len(isles_adc) > 0:
                    matched_data = self.match_histogram_simple(data, isles_adc)
                    matched_adc_data.append(matched_data.flatten()[matched_data.flatten() > 0])
                    
                    # Save matched file
                    matched_img = nib.Nifti1Image(matched_data, img.affine, img.header)
                    output_file = Path(self.output_path) / adc_file.name
                    nib.save(matched_img, str(output_file))
                
                processed += 1
                self.progress['value'] = 30 + (processed / total_files) * 50
            
            # Process HCP DWI files
            for dwi_file in hcp_files['dwi']:
                self.status_label.config(text=f"Processing {dwi_file.name}...")
                self.root.update()
                
                img = nib.load(str(dwi_file))
                data = img.get_fdata()
                # Convert HCP data from m²/s to mm²/s
                median_val = np.median(data[data > 0])
                print(f"Processing HCP DWI: {dwi_file.name}, median before: {median_val}")
                data = data * 1000.0
                print(f"  -> Converted to mm²/s, median after: {np.median(data[data > 0])}")
                
                if len(isles_dwi) > 0:
                    matched_data = self.match_histogram_simple(data, isles_dwi)
                    matched_dwi_data.append(matched_data.flatten()[matched_data.flatten() > 0])
                    
                    # Save matched file
                    matched_img = nib.Nifti1Image(matched_data, img.affine, img.header)
                    output_file = Path(self.output_path) / dwi_file.name
                    nib.save(matched_img, str(output_file))
                
                processed += 1
                self.progress['value'] = 30 + (processed / total_files) * 50
            
            self.progress['value'] = 80
            
            # Plot histograms
            self.status_label.config(text="Generating histograms...")
            self.root.update()
            self.plot_histograms(isles_adc, isles_dwi, matched_adc_data, matched_dwi_data)
            
            self.progress['value'] = 100
            self.status_label.config(text="Processing complete!", foreground="green")
            
            messagebox.showinfo("Success", f"Processed {total_files} files successfully!\nHistograms displayed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error occurred", foreground="red")
        finally:
            self.process_btn.config(state=tk.NORMAL)
    
    def plot_histograms(self, isles_adc, isles_dwi, matched_adc, matched_dwi):
        """Plot comparison histograms"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ADC histograms
        if len(isles_adc) > 0 and len(matched_adc) > 0:
            matched_adc_combined = np.concatenate(matched_adc)
            
            axes[0, 0].hist(isles_adc, bins=100, alpha=0.7, label='ISLES (reference)', color='blue', density=True)
            axes[0, 0].set_title('ISLES ADC Distribution')
            axes[0, 0].set_xlabel('Intensity')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            
            axes[0, 1].hist(matched_adc_combined, bins=100, alpha=0.7, label='HCP (matched)', color='orange', density=True)
            axes[0, 1].set_title('HCP ADC Distribution (After Matching)')
            axes[0, 1].set_xlabel('Intensity')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
        
        # DWI histograms
        if len(isles_dwi) > 0 and len(matched_dwi) > 0:
            matched_dwi_combined = np.concatenate(matched_dwi)
            
            axes[1, 0].hist(isles_dwi, bins=100, alpha=0.7, label='ISLES (reference)', color='blue', density=True)
            axes[1, 0].set_title('ISLES DWI Distribution')
            axes[1, 0].set_xlabel('Intensity')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            
            axes[1, 1].hist(matched_dwi_combined, bins=100, alpha=0.7, label='HCP (matched)', color='orange', density=True)
            axes[1, 1].set_title('HCP DWI Distribution (After Matching)')
            axes[1, 1].set_xlabel('Intensity')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


def main():
    root = tk.Tk()
    app = HistogramMatchingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()