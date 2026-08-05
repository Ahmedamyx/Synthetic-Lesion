#!/usr/bin/env python3
"""
Batch Synthetic Lesion Generator for ADC/DWI pairs.

Processes multiple ADC/DWI pairs from a folder:
- Uses ADC as the anatomical reference for lesion placement
- Applies consistent lesions to both ADC and DWI
- ADC: decreased intensity (simulating restricted diffusion)
- DWI: increased intensity (simulating bright lesions)
- Batch processes all matching pairs in a folder
- Each ADC/DWI pair gets its own SynthSeg file from a folder
- Number of lesions = Number of connected components (controlled by distribution)
- Lesion locations can be sampled from probability distribution
"""

import os
import datetime
import numpy as np
import nibabel as nib
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import gaussian_kde
from scipy.ndimage import distance_transform_edt, label as ndi_label, gaussian_filter, zoom, binary_dilation
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import random
import pickle
import glob
import re
import json
from pathlib import Path

# ---------- User dialogs ----------

def ask_for_file(title, filetypes=(("NIfTI files", "*.nii*"), ("All files", "*.*"))):
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return p

def ask_for_integer(title, prompt, initial=1):
    root = tk.Tk(); root.withdraw()
    val = simpledialog.askinteger(title, prompt, initialvalue=initial)
    root.destroy()
    return val

def ask_for_string(title, prompt, initial=""):
    root = tk.Tk(); root.withdraw()
    val = simpledialog.askstring(title, prompt, initialvalue=initial)
    root.destroy()
    return val

def ask_for_directory(title="Select Directory"):
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

# ---------- Enhanced Shape and Distribution Selection Dialog ----------

class EnhancedConfigDialog:
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Lesion Generation Configuration")
        self.root.geometry("900x850")  # Increased height for new option
        self.root.resizable(False, False)
        
        self.shape_mode = None
        self.shape_library_path = None
        self.shape_library = None
        
        # Distribution settings
        self.use_dwi_intensity_dist = tk.BooleanVar(value=False)
        self.use_adc_intensity_dist = tk.BooleanVar(value=False) 
        self.use_surrounding_intensity_dist = tk.BooleanVar(value=False)
        self.use_connected_lesions_dist = tk.BooleanVar(value=False)
        self.use_location_dist = tk.BooleanVar(value=False)  # NEW
        
        self.dwi_intensity_kde_path = None
        self.adc_intensity_kde_path = None
        self.surrounding_intensity_kde_path = None
        self.connected_lesions_kde_path = None
        self.location_kde_path = None  # NEW
        
        self.setup_ui()
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.root.winfo_screenheight() // 2) - (425 // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.root.wait_window()
    
    def setup_ui(self):
        # Create main scrollable frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_frame = tk.Frame(scrollable_frame, bg="#34495E", pady=20)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="Lesion Generation Configuration",
            font=("Arial", 18, "bold"),
            bg="#34495E",
            fg="white"
        )
        title_label.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Configure shape mode and optional distributions",
            font=("Arial", 10),
            bg="#34495E",
            fg="#BDC3C7"
        )
        subtitle.pack()
        
        # Main content
        content_frame = tk.Frame(scrollable_frame, padx=40, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== SECTION 1: Shape Selection =====
        shape_section = tk.LabelFrame(
            content_frame,
            text="Step 1: Select Lesion Shape Mode",
            font=("Arial", 13, "bold"),
            padx=20,
            pady=15,
            bg="#ECF0F1"
        )
        shape_section.pack(fill=tk.X, pady=(0, 15))
        
        # Spherical option
        sphere_frame = tk.Frame(shape_section, bg="#ECF0F1")
        sphere_frame.pack(fill=tk.X, pady=5)
        
        sphere_btn = tk.Button(
            sphere_frame,
            text="Use Spherical Shapes",
            command=self.select_sphere,
            bg="#3498DB",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8,
            cursor="hand2",
            width=25
        )
        sphere_btn.pack(side=tk.LEFT, padx=5)
        
        sphere_label = tk.Label(
            sphere_frame,
            text="Simple spherical lesions using distance transform",
            font=("Arial", 9),
            bg="#ECF0F1",
            fg="#555555"
        )
        sphere_label.pack(side=tk.LEFT, padx=10)
        
        # Custom shapes option
        custom_frame = tk.Frame(shape_section, bg="#ECF0F1")
        custom_frame.pack(fill=tk.X, pady=5)
        
        custom_btn = tk.Button(
            custom_frame,
            text="Import Custom Shapes",
            command=self.select_custom,
            bg="#27AE60",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8,
            cursor="hand2",
            width=25
        )
        custom_btn.pack(side=tk.LEFT, padx=5)
        
        custom_label = tk.Label(
            custom_frame,
            text="Use realistic morphological shapes from library",
            font=("Arial", 9),
            bg="#ECF0F1",
            fg="#555555"
        )
        custom_label.pack(side=tk.LEFT, padx=10)
        
        # Shape status label
        self.shape_status_label = tk.Label(
            shape_section,
            text="⚠ No shape mode selected",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#E74C3C"
        )
        self.shape_status_label.pack(pady=(10, 0))
        
        # ===== SECTION 2: Distribution Options =====
        dist_section = tk.LabelFrame(
            content_frame,
            text="Step 2: Optional Distribution Controls (uncheck to use random values)",
            font=("Arial", 13, "bold"),
            padx=20,
            pady=15,
            bg="#ECF0F1"
        )
        dist_section.pack(fill=tk.X, pady=(0, 15))
        
        # DWI Intensity Distribution
        dwi_frame = tk.Frame(dist_section, bg="#ECF0F1")
        dwi_frame.pack(fill=tk.X, pady=8)
        
        dwi_check = tk.Checkbutton(
            dwi_frame,
            text="DWI Lesion Intensity Multiplier",
            variable=self.use_dwi_intensity_dist,
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            command=self.toggle_dwi_dist
        )
        dwi_check.pack(side=tk.LEFT)
        
        self.dwi_load_btn = tk.Button(
            dwi_frame,
            text="Load KDE",
            command=self.load_dwi_intensity_kde,
            bg="#9B59B6",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.dwi_load_btn.pack(side=tk.LEFT, padx=10)
        
        self.dwi_status = tk.Label(
            dwi_frame,
            text="Not loaded",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#95A5A6"
        )
        self.dwi_status.pack(side=tk.LEFT, padx=5)
        
        dwi_desc = tk.Label(
            dist_section,
            text="   Controls the DWI intensity multiplication factor inside lesions (default: random 1.7-2.2)",
            font=("Arial", 8),
            bg="#ECF0F1",
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        dwi_desc.pack(anchor=tk.W, padx=20)
        
        # ADC Intensity Distribution
        adc_frame = tk.Frame(dist_section, bg="#ECF0F1")
        adc_frame.pack(fill=tk.X, pady=8)
        
        adc_check = tk.Checkbutton(
            adc_frame,
            text="ADC Lesion Intensity Multiplier",
            variable=self.use_adc_intensity_dist,
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            command=self.toggle_adc_dist
        )
        adc_check.pack(side=tk.LEFT)
        
        self.adc_load_btn = tk.Button(
            adc_frame,
            text="Load KDE",
            command=self.load_adc_intensity_kde,
            bg="#3498DB",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.adc_load_btn.pack(side=tk.LEFT, padx=10)
        
        self.adc_status = tk.Label(
            adc_frame,
            text="Not loaded",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#95A5A6"
        )
        self.adc_status.pack(side=tk.LEFT, padx=5)
        
        adc_desc = tk.Label(
            dist_section,
            text="   Controls the ADC intensity multiplication factor inside lesions (default: random 0.5-0.9)",
            font=("Arial", 8),
            bg="#ECF0F1",
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        adc_desc.pack(anchor=tk.W, padx=20)
        
        # Surrounding Tissue Intensity Distribution
        surr_frame = tk.Frame(dist_section, bg="#ECF0F1")
        surr_frame.pack(fill=tk.X, pady=8)
        
        surr_check = tk.Checkbutton(
            surr_frame,
            text="Surrounding Tissue Intensity",
            variable=self.use_surrounding_intensity_dist,
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            command=self.toggle_surrounding_dist
        )
        surr_check.pack(side=tk.LEFT)
        
        self.surr_load_btn = tk.Button(
            surr_frame,
            text="Load KDE",
            command=self.load_surrounding_intensity_kde,
            bg="#E67E22",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.surr_load_btn.pack(side=tk.LEFT, padx=10)
        
        self.surr_status = tk.Label(
            surr_frame,
            text="Not loaded",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#95A5A6"
        )
        self.surr_status.pack(side=tk.LEFT, padx=5)
        
        surr_desc = tk.Label(
            dist_section,
            text="   Controls the intensity added to tissue surrounding lesions (default: random 0-20% increase)",
            font=("Arial", 8),
            bg="#ECF0F1",
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        surr_desc.pack(anchor=tk.W, padx=20)
        
        # Connected Lesions Distribution
        conn_frame = tk.Frame(dist_section, bg="#ECF0F1")
        conn_frame.pack(fill=tk.X, pady=8)
        
        conn_check = tk.Checkbutton(
            conn_frame,
            text="Number of Connected Lesions",
            variable=self.use_connected_lesions_dist,
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            command=self.toggle_connected_dist
        )
        conn_check.pack(side=tk.LEFT)
        
        self.conn_load_btn = tk.Button(
            conn_frame,
            text="Load KDE",
            command=self.load_connected_lesions_kde,
            bg="#16A085",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.conn_load_btn.pack(side=tk.LEFT, padx=10)
        
        self.conn_status = tk.Label(
            conn_frame,
            text="Not loaded",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#95A5A6"
        )
        self.conn_status.pack(side=tk.LEFT, padx=5)
        
        conn_desc = tk.Label(
            dist_section,
            text="   Controls how many TOTAL lesions are placed (also affects grouping)",
            font=("Arial", 8),
            bg="#ECF0F1",
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        conn_desc.pack(anchor=tk.W, padx=20)
        
        # ===== NEW: Lesion Location Distribution =====
        loc_frame = tk.Frame(dist_section, bg="#ECF0F1")
        loc_frame.pack(fill=tk.X, pady=8)
        
        loc_check = tk.Checkbutton(
            loc_frame,
            text="Lesion Location Probability",
            variable=self.use_location_dist,
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            command=self.toggle_location_dist
        )
        loc_check.pack(side=tk.LEFT)
        
        self.loc_load_btn = tk.Button(
            loc_frame,
            text="Load KDE",
            command=self.load_location_kde,
            bg="#E74C3C",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.loc_load_btn.pack(side=tk.LEFT, padx=10)
        
        self.loc_status = tk.Label(
            loc_frame,
            text="Not loaded",
            font=("Arial", 9, "italic"),
            bg="#ECF0F1",
            fg="#95A5A6"
        )
        self.loc_status.pack(side=tk.LEFT, padx=5)
        
        loc_desc = tk.Label(
            dist_section,
            text="   Controls WHERE lesions are placed based on real lesion center distribution (default: random)",
            font=("Arial", 8),
            bg="#ECF0F1",
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        loc_desc.pack(anchor=tk.W, padx=20)
        
        # ===== SECTION 3: Action Buttons =====
        button_frame = tk.Frame(content_frame)
        button_frame.pack(pady=20)
        
        proceed_btn = tk.Button(
            button_frame,
            text="Proceed with Configuration",
            command=self.proceed,
            bg="#2ECC71",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=30,
            pady=12,
            cursor="hand2"
        )
        proceed_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel,
            bg="#95A5A6",
            fg="white",
            font=("Arial", 10),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def select_sphere(self):
        self.shape_mode = 'sphere'
        self.shape_status_label.config(
            text="✓ Spherical shapes selected",
            fg="#27AE60"
        )
    
    def select_custom(self):
        filepath = filedialog.askopenfilename(
            title="Select Shape Library File (shape_library.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'rb') as f:
                library = pickle.load(f)
            
            if 'shapes' not in library or 'metrics' not in library:
                messagebox.showerror(
                    "Invalid Library",
                    "The selected file is not a valid shape library.\n"
                    "Expected keys: 'shapes', 'metrics'"
                )
                return
            
            num_shapes = len(library['shapes'])
            messagebox.showinfo(
                "Library Loaded",
                f"Successfully loaded shape library!\n\n"
                f"Number of shapes: {num_shapes}\n"
                f"File: {os.path.basename(filepath)}"
            )
            
            self.shape_mode = 'custom'
            self.shape_library_path = filepath
            self.shape_library = library
            self.shape_status_label.config(
                text=f"✓ Custom shapes loaded ({num_shapes} shapes)",
                fg="#27AE60"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Load Error",
                f"Failed to load shape library:\n\n{str(e)}"
            )
    
    def toggle_dwi_dist(self):
        if self.use_dwi_intensity_dist.get():
            self.dwi_load_btn.config(state=tk.NORMAL)
        else:
            self.dwi_load_btn.config(state=tk.DISABLED)
            self.dwi_intensity_kde_path = None
            self.dwi_status.config(text="Not loaded", fg="#95A5A6")
    
    def toggle_adc_dist(self):
        if self.use_adc_intensity_dist.get():
            self.adc_load_btn.config(state=tk.NORMAL)
        else:
            self.adc_load_btn.config(state=tk.DISABLED)
            self.adc_intensity_kde_path = None
            self.adc_status.config(text="Not loaded", fg="#95A5A6")
    
    def toggle_surrounding_dist(self):
        if self.use_surrounding_intensity_dist.get():
            self.surr_load_btn.config(state=tk.NORMAL)
        else:
            self.surr_load_btn.config(state=tk.DISABLED)
            self.surrounding_intensity_kde_path = None
            self.surr_status.config(text="Not loaded", fg="#95A5A6")
    
    def toggle_connected_dist(self):
        if self.use_connected_lesions_dist.get():
            self.conn_load_btn.config(state=tk.NORMAL)
        else:
            self.conn_load_btn.config(state=tk.DISABLED)
            self.connected_lesions_kde_path = None
            self.conn_status.config(text="Not loaded", fg="#95A5A6")
    
    def toggle_location_dist(self):
        if self.use_location_dist.get():
            self.loc_load_btn.config(state=tk.NORMAL)
        else:
            self.loc_load_btn.config(state=tk.DISABLED)
            self.location_kde_path = None
            self.loc_status.config(text="Not loaded", fg="#95A5A6")
    
    def load_dwi_intensity_kde(self):
        filepath = filedialog.askopenfilename(
            title="Select DWI Intensity KDE (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.dwi_intensity_kde_path = filepath
            self.dwi_status.config(
                text=f"✓ {os.path.basename(filepath)}",
                fg="#27AE60"
            )
    
    def load_adc_intensity_kde(self):
        filepath = filedialog.askopenfilename(
            title="Select ADC Intensity KDE (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.adc_intensity_kde_path = filepath
            self.adc_status.config(
                text=f"✓ {os.path.basename(filepath)}",
                fg="#27AE60"
            )
    
    def load_surrounding_intensity_kde(self):
        filepath = filedialog.askopenfilename(
            title="Select Surrounding Intensity KDE (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.surrounding_intensity_kde_path = filepath
            self.surr_status.config(
                text=f"✓ {os.path.basename(filepath)}",
                fg="#27AE60"
            )
    
    def load_connected_lesions_kde(self):
        filepath = filedialog.askopenfilename(
            title="Select Connected Lesions KDE (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.connected_lesions_kde_path = filepath
            self.conn_status.config(
                text=f"✓ {os.path.basename(filepath)}",
                fg="#27AE60"
            )
    
    def load_location_kde(self):
        filepath = filedialog.askopenfilename(
            title="Select Lesion Location KDE (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.location_kde_path = filepath
            self.loc_status.config(
                text=f"✓ {os.path.basename(filepath)}",
                fg="#27AE60"
            )
    
    def proceed(self):
        if self.shape_mode is None:
            messagebox.showerror(
                "Shape Mode Required",
                "Please select a shape mode (Spherical or Custom) before proceeding."
            )
            return
        
        # Validate that if distributions are checked, they have files loaded
        if self.use_dwi_intensity_dist.get() and not self.dwi_intensity_kde_path:
            messagebox.showerror(
                "Missing KDE",
                "DWI intensity distribution is enabled but no KDE file is loaded."
            )
            return
        
        if self.use_adc_intensity_dist.get() and not self.adc_intensity_kde_path:
            messagebox.showerror(
                "Missing KDE",
                "ADC intensity distribution is enabled but no KDE file is loaded."
            )
            return
        
        if self.use_surrounding_intensity_dist.get() and not self.surrounding_intensity_kde_path:
            messagebox.showerror(
                "Missing KDE",
                "Surrounding intensity distribution is enabled but no KDE file is loaded."
            )
            return
        
        if self.use_connected_lesions_dist.get() and not self.connected_lesions_kde_path:
            messagebox.showerror(
                "Missing KDE",
                "Connected lesions distribution is enabled but no KDE file is loaded."
            )
            return
        
        if self.use_location_dist.get() and not self.location_kde_path:
            messagebox.showerror(
                "Missing KDE",
                "Lesion location distribution is enabled but no KDE file is loaded."
            )
            return
        
        self.root.destroy()
    
    def cancel(self):
        self.shape_mode = None
        self.root.destroy()
    
    def get_result(self):
        return {
            'mode': self.shape_mode,
            'library_path': self.shape_library_path,
            'library': self.shape_library,
            'use_dwi_intensity_dist': self.use_dwi_intensity_dist.get(),
            'use_adc_intensity_dist': self.use_adc_intensity_dist.get(),
            'use_surrounding_intensity_dist': self.use_surrounding_intensity_dist.get(),
            'use_connected_lesions_dist': self.use_connected_lesions_dist.get(),
            'use_location_dist': self.use_location_dist.get(),  # NEW
            'dwi_intensity_kde_path': self.dwi_intensity_kde_path,
            'adc_intensity_kde_path': self.adc_intensity_kde_path,
            'surrounding_intensity_kde_path': self.surrounding_intensity_kde_path,
            'connected_lesions_kde_path': self.connected_lesions_kde_path,
            'location_kde_path': self.location_kde_path  # NEW
        }

# ---------- KDE sampler ----------

def sample_from_saved_kde(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    dataset = None
    bw = None

    if 'dataset' in data.files:
        dataset = data['dataset']
    elif 'values' in data.files:
        dataset = data['values']
    else:
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.size > 1:
                dataset = arr
                break
    for k in ('bw_method', 'bw', 'bw_factor', 'bw_method_factor'):
        if k in data.files:
            try: bw = float(data[k]); break
            except: continue
    if dataset is None:
        raise ValueError("Could not detect KDE dataset in NPZ.")

    dataset = np.asarray(dataset).ravel()
    kde = gaussian_kde(dataset)
    if bw is not None:
        try: kde.set_bandwidth(kde.factor * float(bw))
        except: pass

    def sampler(n=1):
        return kde.resample(n).reshape(-1)
    return sampler

# ---------- Location Sampler (NEW) ----------

def sample_from_location_kde(npz_path):
    """
    Load lesion location KDE and create a sampler.
    The KDE contains normalized lesion center coordinates [0, 1] in 3D.
    Returns a function that samples normalized coordinates.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Load dataset (should be shape [3, n] for 3D coordinates)
    dataset = None
    if 'dataset' in data.files:
        dataset = data['dataset']
    else:
        raise ValueError("No 'dataset' found in location KDE NPZ file")
    
    # Get bandwidth
    bw = None
    for k in ('bw_method', 'bw', 'bw_factor', 'bw_method_factor'):
        if k in data.files:
            try: 
                bw = float(data[k])
                break
            except: 
                continue
    
    # Check dataset shape
    if dataset.ndim != 2 or dataset.shape[0] != 3:
        print(f"Warning: Location dataset shape is {dataset.shape}, expected (3, n)")
        # Try to reshape if it's 1D
        if dataset.ndim == 1:
            if len(dataset) % 3 == 0:
                dataset = dataset.reshape(3, -1)
            else:
                raise ValueError(f"1D dataset length {len(dataset)} not divisible by 3")
        else:
            raise ValueError(f"Unexpected dataset shape: {dataset.shape}")
    
    # Create KDE
    kde = gaussian_kde(dataset, bw_method=bw)
    
    def sampler(n=1):
        """
        Sample n lesion locations.
        Returns array of shape (n, 3) with normalized coordinates [0, 1].
        """
        samples = kde.resample(n)
        # Ensure samples are within [0, 1] bounds
        samples = np.clip(samples, 0.0, 1.0)
        return samples.T  # Return (n, 3) shape
    
    return sampler, dataset.shape[1]  # Return sampler and number of lesions in original data

# ---------- Location-Aware Seed Selection (NEW) ----------

def choose_seed_from_location_distribution(allowed_mask, location_sampler, max_attempts=100):
    """
    Choose a seed location using lesion location probability distribution.
    Samples from KDE and finds nearest valid voxel in allowed_mask.
    
    Args:
        allowed_mask: Boolean mask of valid voxels
        location_sampler: Function that returns normalized coordinates
        max_attempts: Maximum number of sampling attempts
    
    Returns:
        Tuple (x, y, z) voxel coordinates or None if failed
    """
    if location_sampler is None:
        return choose_random_seed(allowed_mask)
    
    # Get image shape
    shape = allowed_mask.shape
    
    for attempt in range(max_attempts):
        # Sample normalized coordinates from KDE
        normalized_coords = location_sampler(1)[0]  # Shape (3,)
        
        # Convert to voxel coordinates
        voxel_coords = [
            int(normalized_coords[i] * (shape[i] - 1))
            for i in range(3)
        ]
        
        # Ensure within bounds
        voxel_coords = [
            max(0, min(voxel_coords[i], shape[i] - 1))
            for i in range(3)
        ]
        
        # Check if this voxel is allowed
        if allowed_mask[voxel_coords[0], voxel_coords[1], voxel_coords[2]]:
            return tuple(voxel_coords)
        
        # If not allowed, try to find nearest allowed voxel
        # Get coordinates of all allowed voxels
        allowed_coords = np.argwhere(allowed_mask)
        if len(allowed_coords) == 0:
            return None
        
        # Calculate distances
        distances = np.linalg.norm(allowed_coords - voxel_coords, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_coord = allowed_coords[nearest_idx]
        
        # If nearest is close enough (within 20 voxels), use it
        if distances[nearest_idx] <= 20:
            return tuple(nearest_coord.astype(int))
    
    # Fallback to random seed if all attempts fail
    print("  Warning: Could not find valid location from distribution, using random seed")
    return choose_random_seed(allowed_mask)

# ---------- File matching utilities ----------

def find_adc_dwi_pairs(folder):
    """
    Find matching ADC/DWI pairs in a folder.
    Returns list of tuples: [(adc_path1, dwi_path1), (adc_path2, dwi_path2), ...]
    """
    # Get all NIfTI files
    nifti_files = glob.glob(os.path.join(folder, '*.nii')) + glob.glob(os.path.join(folder, '*.nii.gz'))
    
    # Separate ADC and DWI files
    adc_files = []
    dwi_files = []
    
    for file in nifti_files:
        filename = os.path.basename(file).lower()
        
        # Check for ADC files
        if ('adc' in filename or 'ADC' in filename) and 'lesion' not in filename:
            adc_files.append(file)
        
        # Check for DWI files
        elif ('dwi' in filename or 'DWI' in filename or 'b1000' in filename) and 'lesion' not in filename:
            dwi_files.append(file)
    
    print(f"Found {len(adc_files)} ADC files and {len(dwi_files)} DWI files")
    
    # Try to match files based on naming patterns
    pairs = []
    
    # Method 1: Match by common prefix (e.g., sub-001_adc.nii.gz -> sub-001_dwi.nii.gz)
    for adc_file in adc_files:
        adc_name = os.path.basename(adc_file)
        
        # Remove ADC suffix and extension
        base_name = re.sub(r'[._-]?adc[._-]?', '', adc_name, flags=re.IGNORECASE)
        base_name = re.sub(r'\.nii(\.gz)?$', '', base_name)
        
        # Look for matching DWI
        for dwi_file in dwi_files:
            dwi_name = os.path.basename(dwi_file).lower()
            
            # Check if DWI filename contains the base name
            if base_name.lower() in dwi_name:
                pairs.append((adc_file, dwi_file))
                break
    
    # Method 2: Match by numeric pattern (e.g., ADC1 -> DWI1)
    if len(pairs) < len(adc_files):
        for adc_file in adc_files:
            if any(adc_file in pair for pair in pairs):
                continue  # Already matched
                
            # Extract numbers from filename
            numbers = re.findall(r'\d+', os.path.basename(adc_file))
            if numbers:
                for dwi_file in dwi_files:
                    if any(dwi_file in pair for pair in pairs):
                        continue  # Already matched
                    
                    dwi_numbers = re.findall(r'\d+', os.path.basename(dwi_file))
                    if dwi_numbers and numbers[-1] == dwi_numbers[-1]:
                        pairs.append((adc_file, dwi_file))
                        break
    
    # Method 3: Manual pairing by index if same number of files
    if len(adc_files) == len(dwi_files) and len(pairs) < len(adc_files):
        adc_files.sort()
        dwi_files.sort()
        pairs = list(zip(adc_files, dwi_files))
    
    print(f"Matched {len(pairs)} ADC/DWI pairs")
    return pairs

def match_synthseg_files(adc_dwi_pairs, synthseg_folder):
    """
    Match SynthSeg files to ADC/DWI pairs based on naming patterns.
    Returns list of tuples: [(adc_path, dwi_path, synthseg_path), ...]
    """
    matched_triplets = []
    
    # Get all SynthSeg files
    synthseg_files = glob.glob(os.path.join(synthseg_folder, '*.nii')) + \
                     glob.glob(os.path.join(synthseg_folder, '*.nii.gz'))
    
    print(f"Found {len(synthseg_files)} SynthSeg files in folder")
    
    for adc_path, dwi_path in adc_dwi_pairs:
        adc_name = os.path.basename(adc_path)
        
        # Try different matching strategies
        synthseg_match = None
        
        # Strategy 1: Match by common base name
        base_name = re.sub(r'[._-]?adc[._-]?', '', adc_name, flags=re.IGNORECASE)
        base_name = re.sub(r'\.nii(\.gz)?$', '', base_name)
        base_name = base_name.strip('._-')
        
        # Strategy 2: Extract numbers
        adc_numbers = re.findall(r'\d+', adc_name)
        
        for synthseg_file in synthseg_files:
            synthseg_name = os.path.basename(synthseg_file).lower()
            
            # Check for common patterns
            if ('synthseg' in synthseg_name or 'seg' in synthseg_name or 
                'label' in synthseg_name or 'parc' in synthseg_name):
                
                # Try matching by base name
                if base_name.lower() in synthseg_name.lower():
                    synthseg_match = synthseg_file
                    break
                
                # Try matching by numbers
                if adc_numbers:
                    synthseg_numbers = re.findall(r'\d+', synthseg_name)
                    if synthseg_numbers and adc_numbers[-1] == synthseg_numbers[-1]:
                        synthseg_match = synthseg_file
                        break
        
        if synthseg_match:
            matched_triplets.append((adc_path, dwi_path, synthseg_match))
        else:
            print(f"Warning: No SynthSeg file found for {adc_name}")
    
    print(f"Matched {len(matched_triplets)} ADC/DWI/SynthSeg triplets")
    return matched_triplets

def select_triplets_to_process(triplets):
    """
    Let user select which triplets to process.
    Returns list of selected triplets.
    """
    if not triplets:
        return []
    
    # Create selection dialog
    root = tk.Tk()
    root.title("Select Cases to Process")
    root.geometry("900x500")
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (450 // 2)
    y = (root.winfo_screenheight() // 2) - (250 // 2)
    root.geometry(f"+{x}+{y}")
    
    # Create selection list
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    label = tk.Label(frame, text="Select cases to process:", font=("Arial", 12))
    label.pack(pady=(0, 10))
    
    # Create listbox with checkboxes
    listbox_frame = tk.Frame(frame)
    listbox_frame.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(listbox_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, 
                         yscrollcommand=scrollbar.set, font=("Arial", 10))
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=listbox.yview)
    
    # Add triplets to listbox
    for i, (adc_path, dwi_path, synthseg_path) in enumerate(triplets):
        adc_name = os.path.basename(adc_path)
        dwi_name = os.path.basename(dwi_path)
        synthseg_name = os.path.basename(synthseg_path)
        listbox.insert(tk.END, f"Case {i+1}: {adc_name} ↔ {dwi_name} ↔ {synthseg_name}")
        # Select all by default
        listbox.select_set(i)
    
    selected_indices = []
    
    def on_ok():
        nonlocal selected_indices
        selected_indices = listbox.curselection()
        root.destroy()
    
    def on_select_all():
        listbox.select_set(0, tk.END)
    
    def on_select_none():
        listbox.selection_clear(0, tk.END)
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    select_all_btn = tk.Button(button_frame, text="Select All", command=on_select_all)
    select_all_btn.pack(side=tk.LEFT, padx=5)
    
    select_none_btn = tk.Button(button_frame, text="Select None", command=on_select_none)
    select_none_btn.pack(side=tk.LEFT, padx=5)
    
    ok_btn = tk.Button(button_frame, text="Process Selected", command=on_ok, bg="#2ECC71", fg="white")
    ok_btn.pack(side=tk.LEFT, padx=20)
    
    cancel_btn = tk.Button(button_frame, text="Cancel", command=root.destroy, bg="#95A5A6", fg="white")
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    root.mainloop()
    
    # Return selected triplets
    if selected_indices:
        return [triplets[i] for i in selected_indices]
    return []

# ---------- Utility functions ----------

def convert_cm3_to_voxels(volume_cm3, header):
    zooms = header.get_zooms()[:3]  # Only spatial dimensions
    voxel_mm3 = np.prod(zooms)
    target_mm3 = float(volume_cm3) * 1000.0
    n_vox = max(1, int(round(target_mm3 / voxel_mm3)))
    return n_vox

def choose_random_seed(allowed_mask):
    coords = np.column_stack(np.nonzero(allowed_mask))
    if coords.shape[0] == 0: return None
    idx = random.randrange(coords.shape[0])
    return tuple(int(v) for v in coords[idx])

def grow_spherical_lesion(seed, allowed_mask, target_voxels, voxel_spacing, occupied_mask=None):
    """
    Grow a spherical lesion from seed, strictly respecting target_voxels.
    Returns a boolean lesion mask and actual number of voxels.
    """
    shape = allowed_mask.shape
    comp_labels, ncomps = ndi_label(allowed_mask)
    seed_label = comp_labels[seed]
    if seed_label == 0:
        return np.zeros(shape, bool), 0

    comp_mask = (comp_labels == seed_label)
    
    if occupied_mask is not None:
        comp_mask = comp_mask & (~occupied_mask)
    
    comp_coords = np.argwhere(comp_mask)
    n_comp_vox = comp_coords.shape[0]
    if n_comp_vox == 0:
        return np.zeros(shape, bool), 0

    n_take = min(target_voxels, n_comp_vox)

    if n_take <= 5:
        chosen_indices = np.random.choice(n_comp_vox, n_take, replace=False)
    else:
        distances_mm = np.linalg.norm((comp_coords - np.array(seed)) * np.array(voxel_spacing), axis=1)
        order_idx = np.argsort(distances_mm)
        chosen_indices = order_idx[:n_take]

    lesion_mask = np.zeros(shape, bool)
    for idx in chosen_indices:
        voxel = tuple(comp_coords[idx])
        lesion_mask[voxel] = True

    return lesion_mask, int(lesion_mask.sum())

def place_custom_shape(shape_template, seed, allowed_mask, target_voxels, occupied_mask=None):
    """
    Place a custom shape template at the seed location,
    strictly respecting target_voxels.
    """
    brain_shape = allowed_mask.shape
    shape_coords = np.argwhere(shape_template > 0.5)
    if len(shape_coords) == 0:
        return np.zeros(brain_shape, bool), 0

    current_volume = len(shape_coords)
    scale_factor = (target_voxels / current_volume) ** (1/3) if target_voxels < current_volume else 1.0
    scale_factor = np.clip(scale_factor, 0.01, 3.0)

    if abs(scale_factor - 1.0) > 0.01:
        try:
            scaled_shape = zoom(shape_template, scale_factor, order=1)
            scaled_shape = (scaled_shape > 0.5).astype(np.float32)
        except:
            scaled_shape = shape_template
    else:
        scaled_shape = shape_template

    shape_coords = np.argwhere(scaled_shape > 0.5)
    if len(shape_coords) == 0:
        return np.zeros(brain_shape, bool), 0

    if len(shape_coords) > target_voxels:
        chosen_idx = np.random.choice(len(shape_coords), target_voxels, replace=False)
        shape_coords = shape_coords[chosen_idx]

    shape_center = shape_coords.mean(axis=0)
    offset = np.array(seed) - shape_center

    lesion_mask = np.zeros(brain_shape, bool)
    for coord in shape_coords:
        new_coord = np.round(coord + offset).astype(int)
        if np.all(new_coord >= 0) and np.all(new_coord < brain_shape):
            if allowed_mask[tuple(new_coord)]:
                if occupied_mask is None or not occupied_mask[tuple(new_coord)]:
                    lesion_mask[tuple(new_coord)] = True

    return lesion_mask, int(lesion_mask.sum())

def modify_surrounding_tissue(lesion_mask, image_data, intensity_factor, radius_mm=5.0, voxel_spacing=(1,1,1)):
    """
    Modify the intensity of tissue surrounding the lesion.
    """
    if image_data is None:
        return
    
    # Create a shell around the lesion
    radius_voxels = [max(1, int(radius_mm / vs)) for vs in voxel_spacing]
    dilated = binary_dilation(lesion_mask, iterations=max(radius_voxels))
    surrounding_shell = dilated & (~lesion_mask)
    
    # Apply intensity modification
    image_data[surrounding_shell] *= intensity_factor

# ---------- Main processing function ----------
# Add this after your imports
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def process_adc_dwi_pair(adc_path, dwi_path, synthseg_path, sampler, config, output_dir, 
                         global_params, case_num, total_cases):
    """
    Process a single ADC/DWI pair with its SynthSeg file.
    """
    print(f"\n{'='*80}")
    print(f"Processing case {case_num}/{total_cases}")
    print(f"ADC: {os.path.basename(adc_path)}")
    print(f"DWI: {os.path.basename(dwi_path)}")
    print(f"SynthSeg: {os.path.basename(synthseg_path)}")
    print(f"{'='*80}")
    
    # Load ADC image (used as anatomical reference)
    adc_img = nib.load(adc_path)
    adc_data = adc_img.get_fdata().astype(np.float32)
    
    # Handle 4D ADC data
    if adc_data.ndim == 4:
        print(f"ADC is 4D with shape {adc_data.shape}, extracting first volume...")
        adc_data = adc_data[..., 0]
    
    # Load DWI image (will be modified)
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata().astype(np.float32)
    
    # Handle 4D DWI data
    if dwi_data.ndim == 4:
        print(f"DWI is 4D with shape {dwi_data.shape}, extracting first volume...")
        dwi_data = dwi_data[..., 0]
    
    # Create copies for editing
    adc_data_edited = adc_data.copy()
    dwi_data_edited = dwi_data.copy()
    
    # Use ADC header for spatial info
    brain_header = adc_img.header
    voxel_spacing = brain_header.get_zooms()[:3]
    
    print(f"ADC shape: {adc_data.shape}, DWI shape: {dwi_data.shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    
    # Load and resample labelmap
    synthseg_img = nib.load(synthseg_path)
    label_data = synthseg_img.get_fdata().astype(np.int32)
    
    zoom_factors = np.array(adc_data.shape) / np.array(label_data.shape)
    label_data = zoom(label_data, zoom_factors, order=0)
    print(f"Resampled labelmap shape: {label_data.shape}")
    
    # Define ventricular labels to avoid
    ventricular_labels = {4, 5, 14, 15, 43, 44, 31, 24, 16, 18}
    
    # Create allowed mask from ADC (brain tissue excluding ventricles)
    allowed_mask = (label_data > 0) & (~np.isin(label_data, list(ventricular_labels)))
    allowed_mask = allowed_mask & (~np.isnan(adc_data)) & (adc_data != 0)
    
    print(f"Allowed mask voxels: {np.sum(allowed_mask)}")
    
    # Initialize combined mask
    combined_mask = np.zeros_like(adc_data, dtype=np.uint8)
    metadata = []
    lesion_idx = 1
    
    # Determine TOTAL number of lesions (same as connected components)
    if config['use_connected_lesions_dist'] and global_params['connected_lesions_sampler']:
        # Sample from distribution to get total number of lesions
        total_lesions = max(1, int(round(global_params['connected_lesions_sampler'](1)[0])))
    else:
        # Random total number of lesions (1-5)
        total_lesions = random.randint(1, 5)
    
    print(f"Generating {total_lesions} lesions")
    
    # Generate lesions - NO CLUSTERING/NO GROUPING
    for lesion_num in range(total_lesions):
        print(f"\n--- Generating lesion {lesion_num+1}/{total_lesions} ---")
        
        # Sample volume
        sampled_cm3 = max(0.001, float(sampler(1)[0]))
        target_vox = convert_cm3_to_voxels(sampled_cm3, brain_header)
        
        # Choose seed location based on configuration
        if config['use_location_dist'] and global_params['location_sampler']:
            seed = choose_seed_from_location_distribution(
                allowed_mask, global_params['location_sampler']
            )
            seed_method = 'location_distribution'
        else:
            seed = choose_random_seed(allowed_mask)
            seed_method = 'random'
        
        if seed is None:
            print(f"  Warning: Could not find seed for lesion {lesion_num+1}, skipping...")
            continue
        
        # Check component size
        comp_labels, _ = ndi_label(allowed_mask)
        comp_label_val = comp_labels[seed]
        if comp_label_val == 0:
            print(f"  Warning: Seed at {seed} is not in allowed mask, skipping...")
            continue
        
        comp_size = int((comp_labels == comp_label_val).sum())
        if comp_size < 6:
            print(f"  Warning: Component too small ({comp_size} voxels), skipping...")
            continue
        
        # Generate lesion based on mode
        if config['mode'] == 'sphere':
            lesion_mask, added_vox = grow_spherical_lesion(
                seed, allowed_mask, target_vox, voxel_spacing, combined_mask
            )
        else:  # custom
            shape_idx = random.randint(0, len(config['library']['shapes']) - 1)
            shape_template = config['library']['shapes'][shape_idx]
            lesion_mask, added_vox = place_custom_shape(
                shape_template, seed, allowed_mask, target_vox, combined_mask
            )
        
        # Ensure lesion doesn't overlap with existing lesions
        lesion_mask = lesion_mask & (combined_mask == 0)
        added_vox_after = int(lesion_mask.sum())
        actual_cm3 = added_vox_after * np.prod(voxel_spacing) / 1000.0
        
        if added_vox_after == 0:
            print(f"  Warning: Lesion has no voxels after overlap check, skipping...")
            continue

        # Add to combined mask
        combined_mask[lesion_mask] = lesion_idx
        
        # Determine intensity factors
        if config['use_dwi_intensity_dist'] and global_params['dwi_intensity_sampler']:
            dwi_factor = float(global_params['dwi_intensity_sampler'](1)[0])
            dwi_factor = np.clip(dwi_factor, 1.0, 5.0)
        else:
            dwi_factor = float(np.random.uniform(1.7, 2.2))
        
        if config['use_adc_intensity_dist'] and global_params['adc_intensity_sampler']:
            adc_factor = float(global_params['adc_intensity_sampler'](1)[0])
            adc_factor = np.clip(adc_factor, 0.1, 0.9)
        else:
            adc_factor = float(np.random.uniform(0.5, 0.9))
        
        if config['use_surrounding_intensity_dist'] and global_params['surrounding_intensity_sampler']:
            surr_factor = float(global_params['surrounding_intensity_sampler'](1)[0])
            surr_factor = np.clip(surr_factor, 1.0, 2.0)
        else:
            surr_factor = float(np.random.uniform(1.0, 1.2))
        
        # Apply intensity modifications
        # ADC: decrease intensity (restricted diffusion)
        adc_data_edited[lesion_mask] *= adc_factor
        modify_surrounding_tissue(lesion_mask, adc_data_edited, surr_factor, 
                                 radius_mm=5.0, voxel_spacing=voxel_spacing)
        
        # DWI: increase intensity (bright lesions)
        dwi_data_edited[lesion_mask] *= dwi_factor
        modify_surrounding_tissue(lesion_mask, dwi_data_edited, surr_factor, 
                                 radius_mm=5.0, voxel_spacing=voxel_spacing)
        
        metadata.append({
            'lesion_index': lesion_idx,
            'seed': seed,
            'seed_method': seed_method,
            'requested_voxels': target_vox,
            'requested_cm3': sampled_cm3,
            'actual_voxels': added_vox_after,
            'actual_cm3': actual_cm3,
            'dwi_intensity_factor': dwi_factor,
            'adc_intensity_factor': adc_factor,
            'surrounding_intensity_factor': surr_factor
        })
        
        lesion_idx += 1
        print(f"  ✓ Lesion placed at {seed}, volume: {actual_cm3:.2f} cm³")
    
    # Create output directory for this case
    case_name = os.path.basename(adc_path).split('.')[0].replace('_adc', '').replace('_ADC', '')
    case_output_dir = os.path.join(output_dir, f"processed_{case_name}")
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Save outputs
    mode_suffix = "spherical" if config['mode'] == 'sphere' else "custom"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save masks
    mask_fname = os.path.join(case_output_dir, f"{case_name}_lesion_mask_{mode_suffix}_{timestamp}.nii.gz")
    nib.save(nib.Nifti1Image(combined_mask.astype(np.uint8), affine=adc_img.affine, header=adc_img.header), mask_fname)
    
    if global_params['make_soft']:
        sigma_vox = tuple(global_params['soft_sigma_mm'] / float(s) for s in voxel_spacing)
        smoothed = gaussian_filter((combined_mask > 0).astype(np.float32), sigma=sigma_vox, mode='constant')
        soft_alpha = smoothed / (smoothed.max() if smoothed.max() > 0 else 1.0)
        soft_fname = os.path.join(case_output_dir, f"{case_name}_lesion_soft_mask_{mode_suffix}_{timestamp}.nii.gz")
        nib.save(nib.Nifti1Image(soft_alpha.astype(np.float32), affine=adc_img.affine, header=adc_img.header), soft_fname)
    
    # Save edited images
    adc_out = os.path.join(case_output_dir, f"{case_name}_adc_with_lesions_{mode_suffix}_{timestamp}.nii.gz")
    nib.save(nib.Nifti1Image(adc_data_edited, affine=adc_img.affine, header=adc_img.header), adc_out)
    
    dwi_out = os.path.join(case_output_dir, f"{case_name}_dwi_with_lesions_{mode_suffix}_{timestamp}.nii.gz")
    nib.save(nib.Nifti1Image(dwi_data_edited, affine=dwi_img.affine, header=dwi_img.header), dwi_out)
    
    # Save metadata
    meta_fname = os.path.join(case_output_dir, f"{case_name}_lesion_metadata_{timestamp}.json")
    with open(meta_fname, 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved outputs to: {case_output_dir}")
    print(f"✓ Generated {len(metadata)} lesions")
    
    return {
        'case_name': case_name,
        'adc_path': adc_path,
        'dwi_path': dwi_path,
        'synthseg_path': synthseg_path,
        'output_dir': case_output_dir,
        'num_lesions': len(metadata),
        'metadata': metadata
    }

# ---------- Main flow ----------

def main():
    # Show enhanced configuration dialog
    print("Opening configuration dialog...")
    dialog = EnhancedConfigDialog()
    config = dialog.get_result()
    
    if config['mode'] is None:
        print("User cancelled.")
        return
    
    print(f"\nSelected mode: {config['mode']}")
    if config['mode'] == 'custom':
        print(f"Loaded {len(config['library']['shapes'])} custom shapes")
    
    # Load KDE for lesion volumes
    print("\nSelect KDE .npz file (lesion volume distribution).")
    kde_path = ask_for_file("Select KDE .npz")
    if not kde_path: return
    
    # Load distribution samplers if enabled
    dwi_intensity_sampler = None
    adc_intensity_sampler = None
    surrounding_intensity_sampler = None
    connected_lesions_sampler = None
    location_sampler = None
    location_original_count = 0
    
    if config['use_dwi_intensity_dist']:
        print(f"Loading DWI intensity distribution from {config['dwi_intensity_kde_path']}")
        dwi_intensity_sampler = sample_from_saved_kde(config['dwi_intensity_kde_path'])
    
    if config['use_adc_intensity_dist']:
        print(f"Loading ADC intensity distribution from {config['adc_intensity_kde_path']}")
        adc_intensity_sampler = sample_from_saved_kde(config['adc_intensity_kde_path'])
    
    if config['use_surrounding_intensity_dist']:
        print(f"Loading surrounding intensity distribution from {config['surrounding_intensity_kde_path']}")
        surrounding_intensity_sampler = sample_from_saved_kde(config['surrounding_intensity_kde_path'])
    
    if config['use_connected_lesions_dist']:
        print(f"Loading connected lesions distribution from {config['connected_lesions_kde_path']}")
        connected_lesions_sampler = sample_from_saved_kde(config['connected_lesions_kde_path'])
    
    if config['use_location_dist']:
        print(f"Loading lesion location distribution from {config['location_kde_path']}")
        try:
            location_sampler, location_original_count = sample_from_location_kde(config['location_kde_path'])
            print(f"  Loaded location distribution with {location_original_count} lesion centers")
        except Exception as e:
            print(f"  Error loading location distribution: {e}")
            print("  Falling back to random location placement")
            config['use_location_dist'] = False
    
    # Select folder with ADC/DWI pairs
    print("\nSelect folder containing ADC and DWI images.")
    data_folder = ask_for_directory("Select ADC/DWI folder")
    if not data_folder: return
    
    # Find ADC/DWI pairs
    print(f"\nSearching for ADC/DWI pairs in: {data_folder}")
    pairs = find_adc_dwi_pairs(data_folder)
    
    if not pairs:
        print("No ADC/DWI pairs found in the specified folder.")
        messagebox.showerror("No Pairs Found", "No ADC/DWI pairs found in the selected folder.")
        return
    
    # Select folder with SynthSeg labelmaps
    print("\nSelect folder containing SynthSeg labelmaps.")
    synthseg_folder = ask_for_directory("Select SynthSeg folder")
    if not synthseg_folder: return
    
    # Match SynthSeg files to ADC/DWI pairs
    print("\nMatching SynthSeg files to ADC/DWI pairs...")
    triplets = match_synthseg_files(pairs, synthseg_folder)
    
    if not triplets:
        print("No matching SynthSeg files found for the selected ADC/DWI pairs.")
        messagebox.showerror("No Matches", "No matching SynthSeg files found for the selected ADC/DWI pairs.")
        return
    
    # Let user select which triplets to process
    selected_triplets = select_triplets_to_process(triplets)
    
    if not selected_triplets:
        print("No triplets selected for processing.")
        return
    
    # Ask about soft mask only
    soft_mask_q = ask_for_string("Soft mask?", "Create soft masks? (y/n)", "y")
    make_soft = (str(soft_mask_q).lower().strip() == 'y')
    soft_sigma_mm = 1.5
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(data_folder, f"synthetic_lesions_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_summary = {
        'timestamp': timestamp,
        'shape_mode': config['mode'],
        'num_cases': len(selected_triplets),
        'create_soft_masks': make_soft,
        'soft_mask_sigma_mm': soft_sigma_mm,
        'use_dwi_intensity_dist': config['use_dwi_intensity_dist'],
        'use_adc_intensity_dist': config['use_adc_intensity_dist'],
        'use_surrounding_intensity_dist': config['use_surrounding_intensity_dist'],
        'use_connected_lesions_dist': config['use_connected_lesions_dist'],
        'use_location_dist': config['use_location_dist'],
        'location_original_lesion_count': int(location_original_count) if location_original_count else 0,
        'selected_triplets': [
            {
                'adc': os.path.basename(adc_path),
                'dwi': os.path.basename(dwi_path),
                'synthseg': os.path.basename(synthseg_path)
            }
            for adc_path, dwi_path, synthseg_path in selected_triplets
        ]
    }
    
    config_file = os.path.join(output_dir, "processing_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Number of cases to process: {len(selected_triplets)}")
    print(f"Shape mode: {config['mode']}")
    print(f"Create soft masks: {make_soft}")
    if config['use_connected_lesions_dist']:
        print(f"Number of lesions: Controlled by connected lesions distribution")
    else:
        print(f"Number of lesions: Random (1-5)")
    if config['use_location_dist']:
        print(f"Lesion locations: Sampled from probability distribution ({location_original_count} real lesions)")
    else:
        print(f"Lesion locations: Random placement")
    print(f"{'='*80}\n")
    
    # Load volume sampler
    sampler = sample_from_saved_kde(kde_path)
    
    # Global parameters
    global_params = {
        'make_soft': make_soft,
        'soft_sigma_mm': soft_sigma_mm,
        'dwi_intensity_sampler': dwi_intensity_sampler,
        'adc_intensity_sampler': adc_intensity_sampler,
        'surrounding_intensity_sampler': surrounding_intensity_sampler,
        'connected_lesions_sampler': connected_lesions_sampler,
        'location_sampler': location_sampler  # NEW
    }
    
    # Process each triplet
    results = []
    for i, (adc_path, dwi_path, synthseg_path) in enumerate(selected_triplets):
        result = process_adc_dwi_pair(
            adc_path, dwi_path, synthseg_path, sampler, config, output_dir,
            global_params, i+1, len(selected_triplets)
        )
        results.append(result)
    
    # Generate summary report
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    total_lesions = sum(r['num_lesions'] for r in results)
    avg_lesions = total_lesions / len(results) if len(results) > 0 else 0
    
    print(f"Total cases processed: {len(results)}")
    print(f"Total lesions generated: {total_lesions}")
    print(f"Average lesions per case: {avg_lesions:.1f}")
    
    # Count how many lesions used location distribution
    if config['use_location_dist']:
        location_based_lesions = 0
        for r in results:
            for lesion in r['metadata']:
                if lesion.get('seed_method') == 'location_distribution':
                    location_based_lesions += 1
        print(f"Lesions placed using location distribution: {location_based_lesions}/{total_lesions} ({location_based_lesions/total_lesions*100:.1f}%)")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Synthetic Lesion Generation Batch Processing Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape mode: {config['mode']}\n")
        f.write(f"Number of cases processed: {len(results)}\n")
        f.write(f"Total lesions generated: {total_lesions}\n")
        f.write(f"Average lesions per case: {avg_lesions:.1f}\n")
        if config['use_location_dist']:
            f.write(f"Lesion locations: Sampled from probability distribution\n")
            f.write(f"  (based on {location_original_count} real lesion centers)\n")
        else:
            f.write(f"Lesion locations: Random placement\n")
        f.write("\n")
        
        f.write("Case Details:\n")
        f.write("-"*60 + "\n")
        for r in results:
            f.write(f"\nCase: {r['case_name']}\n")
            f.write(f"  ADC: {os.path.basename(r['adc_path'])}\n")
            f.write(f"  DWI: {os.path.basename(r['dwi_path'])}\n")
            f.write(f"  SynthSeg: {os.path.basename(r['synthseg_path'])}\n")
            f.write(f"  Lesions generated: {r['num_lesions']}\n")
            f.write(f"  Output directory: {r['output_dir']}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("All outputs saved to:\n")
        f.write(f"{output_dir}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    # Show completion message
    messagebox.showinfo(
        "Processing Complete",
        f"Batch processing complete!\n\n"
        f"Processed {len(results)} cases\n"
        f"Generated {total_lesions} lesions\n"
        f"Average lesions per case: {avg_lesions:.1f}\n"
        f"{'Using location distribution' if config['use_location_dist'] else 'Random locations'}\n\n"
        f"Outputs saved to:\n{output_dir}"
    )

if __name__ == "__main__":
    main()