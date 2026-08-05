import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class MorphologicalShapeGenerator:
    def __init__(self, kde_folder_path, volume_shape=0.26, volume_scale=1664.9):
        """
        Initialize the shape generator with distribution models.
        
        Parameters:
        -----------
        kde_folder_path : str
            Path to folder containing KDE CSV files
        volume_shape : float
            Gamma distribution shape parameter for volume
        volume_scale : float
            Gamma distribution scale parameter for volume
        """
        self.kde_folder = kde_folder_path
        self.volume_shape = volume_shape
        self.volume_scale = volume_scale
        
        # Load KDE distributions
        self.distributions = self.load_kde_distributions()
        print("✓ Loaded KDE distributions")
        
    def load_kde_distributions(self):
        """Load all KDE distributions from CSV files."""
        distributions = {}
        
        kde_files = {
            'convexity': 'convexity_kde.csv',
            'roughness': 'roughness_kde.csv',
            'eccentricity': 'eccentricity_kde.csv',
            'compactness': 'compactness_kde.csv',
            'roundness': 'roundness_kde.csv'
        }
        
        # Try alternative spelling for eccentricity
        if not os.path.exists(os.path.join(self.kde_folder, 'eccentricity_kde.csv')):
            if os.path.exists(os.path.join(self.kde_folder, 'excentricity_kde.csv')):
                kde_files['eccentricity'] = 'excentricity_kde.csv'
        
        for name, filename in kde_files.items():
            filepath = os.path.join(self.kde_folder, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Assuming CSV has 'x' and 'density' columns
                distributions[name] = {
                    'x': df.iloc[:, 0].values,
                    'density': df.iloc[:, 1].values
                }
                print(f"  Loaded {name}: range [{distributions[name]['x'].min():.3f}, {distributions[name]['x'].max():.3f}]")
            else:
                print(f"  Warning: {filename} not found")
        
        return distributions
    
    def sample_from_kde(self, descriptor_name):
        """Sample a value from a KDE distribution."""
        if descriptor_name not in self.distributions:
            raise ValueError(f"Distribution for {descriptor_name} not loaded")
        
        dist = self.distributions[descriptor_name]
        x = dist['x']
        density = dist['density']
        
        # Normalize density to create probability
        prob = density / np.sum(density)
        
        # Sample from the distribution
        idx = np.random.choice(len(x), p=prob)
        # Add some noise within the bin
        if idx < len(x) - 1:
            value = x[idx] + np.random.uniform(0, x[idx+1] - x[idx])
        else:
            value = x[idx]
        
        return value
    
    def sample_volume(self):
        """Sample volume from gamma distribution (in voxels)."""
        volume = np.random.gamma(self.volume_shape, self.volume_scale)
        return max(volume, 10)  # Minimum volume to ensure shape is visible
    
    def create_initial_sphere(self, volume, grid_size=128):
        """Create initial spherical shape based on volume."""
        # Calculate radius from volume
        radius = (3 * volume / (4 * np.pi)) ** (1/3)
        
        # Create grid
        center = grid_size // 2
        x, y, z = np.ogrid[:grid_size, :grid_size, :grid_size]
        
        # Create sphere
        distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
        mask = distance <= radius
        
        return mask.astype(np.float32), center, radius
    
    def compute_eccentricity(self, mask):
        """Compute eccentricity of the shape."""
        coords = np.argwhere(mask > 0.5)
        if len(coords) < 3:
            return 0.0
        
        # Compute covariance matrix
        cov = np.cov(coords.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Eccentricity based on eigenvalues
        if eigenvalues[0] > 0:
            eccentricity = 1 - (eigenvalues[2] / eigenvalues[0])
        else:
            eccentricity = 0.0
        
        return eccentricity
    
    def compute_roughness(self, mask):
        """Compute surface roughness."""
        # Extract surface
        surface = mask.astype(float) - ndimage.binary_erosion(mask).astype(float)
        surface_coords = np.argwhere(surface > 0.5)
        
        if len(surface_coords) < 10:
            return 0.0
        
        # Compute center
        center = surface_coords.mean(axis=0)
        
        # Compute distances from center
        distances = np.linalg.norm(surface_coords - center, axis=1)
        
        # Roughness as coefficient of variation of distances
        if distances.mean() > 0:
            roughness = distances.std() / distances.mean()
        else:
            roughness = 0.0
        
        return roughness
    
    def compute_convexity(self, mask):
        """Compute convexity ratio (lesion area / convex hull area)."""
        coords = np.argwhere(mask > 0.5)
        if len(coords) < 4:
            return 1.0
        
        try:
            hull = ConvexHull(coords)
            hull_volume = hull.volume
            lesion_volume = np.sum(mask)
            
            if hull_volume > 0:
                convexity = lesion_volume / hull_volume
            else:
                convexity = 1.0
        except:
            convexity = 1.0
        
        return convexity
    
    def compute_compactness(self, mask):
        """Compute compactness."""
        volume = np.sum(mask)
        
        # Extract surface
        surface = mask.astype(float) - ndimage.binary_erosion(mask).astype(float)
        surface_area = np.sum(surface)
        
        if surface_area > 0:
            # Compactness = V^(2/3) / A
            compactness = (volume ** (2/3)) / surface_area
        else:
            compactness = 0.0
        
        return compactness
    
    def compute_roundness(self, mask):
        """Compute roundness (sphericity)."""
        volume = np.sum(mask)
        
        # Extract surface
        surface = mask.astype(float) - ndimage.binary_erosion(mask).astype(float)
        surface_area = np.sum(surface)
        
        if surface_area > 0:
            # Roundness based on surface area to volume ratio
            sphere_surface = (36 * np.pi * volume**2) ** (1/3)
            roundness = sphere_surface / surface_area if surface_area > 0 else 0
        else:
            roundness = 0.0
        
        return min(roundness, 1.0)
    
    def adjust_eccentricity(self, mask, target_eccentricity):
        """Adjust shape eccentricity by stretching along principal axis."""
        current_ecc = self.compute_eccentricity(mask)
        
        coords = np.argwhere(mask > 0.5)
        if len(coords) < 3:
            return mask
        
        center = coords.mean(axis=0)
        
        # Compute principal axis
        cov = np.cov(coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Determine stretch factor
        if target_eccentricity > current_ecc:
            stretch_factor = 1 + 0.3 * (target_eccentricity - current_ecc)
        else:
            stretch_factor = 1 - 0.3 * (current_ecc - target_eccentricity)
        
        # Create stretched coordinates
        new_mask = np.zeros_like(mask)
        for coord in coords:
            # Project onto principal axis
            rel_coord = coord - center
            projection = np.dot(rel_coord, principal_axis) * principal_axis
            perpendicular = rel_coord - projection
            
            # Stretch along principal axis
            new_coord = center + stretch_factor * projection + perpendicular
            new_coord = np.round(new_coord).astype(int)
            
            # Check bounds
            if np.all(new_coord >= 0) and np.all(new_coord < mask.shape):
                new_mask[tuple(new_coord)] = 1
        
        return new_mask.astype(np.float32)
    
    def adjust_roughness(self, mask, target_roughness):
        """Adjust surface roughness by adding perturbations."""
        current_roughness = self.compute_roughness(mask)
        
        # Extract surface
        surface = mask.astype(float) - ndimage.binary_erosion(mask).astype(float)
        surface_coords = np.argwhere(surface > 0.5)
        
        if len(surface_coords) < 10:
            return mask
        
        center = surface_coords.mean(axis=0)
        
        # Add random perturbations to surface
        new_mask = mask.copy()
        
        if target_roughness > current_roughness:
            # Add bumps
            perturbation_strength = 0.5 * (target_roughness - current_roughness) / (target_roughness + 1e-6)
            num_perturbations = max(int(len(surface_coords) * 0.1), 10)
            
            for _ in range(num_perturbations):
                idx = np.random.randint(0, len(surface_coords))
                coord = surface_coords[idx]
                direction = coord - center
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                
                # Add small bump outward
                bump_size = int(3 * perturbation_strength * 10)
                for i in range(-bump_size, bump_size+1):
                    for j in range(-bump_size, bump_size+1):
                        for k in range(-bump_size, bump_size+1):
                            if i**2 + j**2 + k**2 <= bump_size**2:
                                new_coord = coord + np.array([i, j, k]) + direction * 2
                                new_coord = np.round(new_coord).astype(int)
                                if np.all(new_coord >= 0) and np.all(new_coord < mask.shape):
                                    new_mask[tuple(new_coord)] = 1
        else:
            # Smooth surface
            new_mask = ndimage.gaussian_filter(new_mask, sigma=1.0)
            new_mask = (new_mask > 0.5).astype(np.float32)
        
        return new_mask
    
    def adjust_convexity(self, mask, target_convexity):
        """Adjust convexity by adding concave regions."""
        current_convexity = self.compute_convexity(mask)
        
        coords = np.argwhere(mask > 0.5)
        if len(coords) < 10:
            return mask
        
        center = coords.mean(axis=0)
        
        new_mask = mask.copy()
        
        if target_convexity < current_convexity:
            # Add concave regions (indentations)
            num_indentations = max(int(5 * (1 - target_convexity)), 1)
            
            for _ in range(num_indentations):
                # Random point on surface
                surface = mask.astype(float) - ndimage.binary_erosion(mask).astype(float)
                surface_coords = np.argwhere(surface > 0.5)
                
                if len(surface_coords) > 0:
                    idx = np.random.randint(0, len(surface_coords))
                    indent_center = surface_coords[idx]
                    
                    # Create indentation (remove sphere)
                    indent_radius = np.random.uniform(3, 8)
                    x, y, z = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
                    distance = np.sqrt((x - indent_center[0])**2 + 
                                     (y - indent_center[1])**2 + 
                                     (z - indent_center[2])**2)
                    indent_mask = distance <= indent_radius
                    new_mask[indent_mask] = 0
        
        return new_mask
    
    def visualize_shape(self, mask, title="Shape", ax=None):
        """Visualize 3D shape."""
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Extract surface using marching cubes
        try:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           cmap='viridis', alpha=0.8, edgecolor='none')
        except:
            # Fallback: plot voxels
            coords = np.argwhere(mask > 0.5)
            if len(coords) > 0:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                          c='blue', alpha=0.1, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Equal aspect ratio
        max_range = np.array([mask.shape[0], mask.shape[1], mask.shape[2]]).max() / 2.0
        mid_x = mask.shape[0] / 2.0
        mid_y = mask.shape[1] / 2.0
        mid_z = mask.shape[2] / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return ax
    def sample_safe(self, descriptor_name, min_value=1e-3):
        """Sample from KDE and ensure value is above a minimum threshold."""
        value = self.sample_from_kde(descriptor_name)
        if value < min_value:
            # Resample until we get a valid value
            for _ in range(10):
                value = self.sample_from_kde(descriptor_name)
                if value >= min_value:
                    break
            else:
                # Fallback to minimum
                value = min_value
        return value
    def generate_shape(self, grid_size=128, visualize=True, max_iters=50, tol=0.05):
        """
        Generate a single shape matching all morphological distributions using iterative adjustments.
        
        Parameters:
        -----------
        grid_size : int
            Size of the 3D grid
        visualize : bool
            Whether to show intermediate visualizations
        max_iters : int
            Maximum number of adjustment iterations
        tol : float
            Tolerance for metric vs target (fractional, e.g., 0.05 = 5%)
        
        Returns:
        --------
        mask : ndarray
            3D binary mask of the generated shape
        metrics : dict
            Dictionary of computed morphological metrics
        """
        print("\n" + "="*60)
        print("GENERATING NEW SHAPE (ITERATIVE)")
        print("="*60)

        # Sample target values from distributions
        target_volume = max(self.sample_volume(), 10)  # Minimum volume
        target_eccentricity = self.sample_safe('eccentricity')
        target_roughness = self.sample_safe('roughness')
        target_convexity = self.sample_safe('convexity')
        target_compactness = self.sample_safe('compactness')
        target_roundness = self.sample_safe('roundness')

        # Safety check: skip impossible targets
        if target_volume <= 0 or target_eccentricity < 0 or target_roughness < 0 or target_convexity <= 0:
            print("⚠ Skipping shape due to invalid target parameters")
            return None, None

        targets = {
            'volume': target_volume,
            'eccentricity': target_eccentricity,
            'roughness': target_roughness,
            'convexity': target_convexity,
            'compactness': target_compactness,
            'roundness': target_roundness
        }

        print(f"\nTarget values:")
        for k, v in targets.items():
            print(f"  {k.capitalize()}: {v:.3f}")

        # Step 1: Initial sphere
        mask, center, radius = self.create_initial_sphere(target_volume, grid_size)

        for iteration in range(max_iters):
            # Iteratively adjust metrics
            mask = self.adjust_eccentricity(mask, target_eccentricity)
            mask = self.adjust_roughness(mask, target_roughness)
            mask = self.adjust_convexity(mask, target_convexity)

            # Rescale volume at each iteration
            current_volume = np.sum(mask)
            scale_factor = (target_volume / current_volume) ** (1/3)
            coords = np.argwhere(mask > 0.5)
            new_mask = np.zeros_like(mask)
            center = coords.mean(axis=0)
            for coord in coords:
                new_coord = center + (coord - center) * scale_factor
                new_coord = np.round(new_coord).astype(int)
                if np.all(new_coord >= 0) and np.all(new_coord < mask.shape):
                    new_mask[tuple(new_coord)] = 1
            mask = new_mask.astype(np.float32)

            # Compute current metrics
            metrics = {
                'volume': np.sum(mask),
                'eccentricity': self.compute_eccentricity(mask),
                'roughness': self.compute_roughness(mask),
                'convexity': self.compute_convexity(mask),
                'compactness': self.compute_compactness(mask),
                'roundness': self.compute_roundness(mask)
            }

            # Check if all metrics are within tolerance
            within_tol = all(
                abs(metrics[k] - targets[k]) / max(targets[k], 1e-6) < tol
                for k in targets
            )
            if within_tol:
                print(f"\n✓ Target metrics reached after {iteration+1} iterations")
                break

        # Visualization of final shape
        if visualize:
            self.visualize_shape(mask, title="Final Iterative Shape")
            plt.show()

        # Add target metrics to output
        metrics.update({f"target_{k}": v for k, v in targets.items()})

        return mask, metrics

    
    def generate_multiple_shapes(self, num_shapes=5, grid_size=128, save_path=None):
        """
        Generate multiple shapes and optionally save them.
        
        Parameters:
        -----------
        num_shapes : int
            Number of shapes to generate
        grid_size : int
            Size of 3D grid
        save_path : str or None
            Path to save generated shapes as NIfTI files
        
        Returns:
        --------
        shapes : list of ndarrays
            List of generated shape masks
        all_metrics : list of dicts
            List of metrics for each shape
        """
        shapes = []
        all_metrics = []
        
        i = 0
        while i < num_shapes:
            mask, metrics = self.generate_shape(grid_size=grid_size, visualize=True)
            
            if mask is None:
                print("Shape skipped due to invalid parameters. Retrying...")
                continue  # Retry without incrementing i

            shapes.append(mask)
            all_metrics.append(metrics)

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                img = nib.Nifti1Image(mask, np.eye(4))
                filename = os.path.join(save_path, f"generated_shape_{i+1:03d}.nii.gz")
                nib.save(img, filename)
                print(f"\nSaved shape to: {filename}")

            i += 1

        
        # Summary statistics
        print("\n\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        
        metrics_df = pd.DataFrame(all_metrics)
        print("\nGenerated vs Target Metrics:")
        print(metrics_df[['volume', 'target_volume', 'eccentricity', 'target_eccentricity',
                         'roughness', 'target_roughness', 'convexity', 'target_convexity']].to_string())
        
        return shapes, all_metrics


class ShapeGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Morphological Shape Generator")
        self.root.geometry("800x900")
        self.root.resizable(True, True)
        
        self.kde_folder = None
        self.save_folder = None
        self.generator = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2C3E50", pady=20)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="3D Morphological Shape Generator",
            font=("Arial", 18, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Generate synthetic 3D shapes matching morphological distributions",
            font=("Arial", 10),
            bg="#2C3E50",
            fg="#BDC3C7"
        )
        subtitle_label.pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section 1: Load KDE Files
        section1 = tk.LabelFrame(
            main_frame,
            text="Step 1: Load KDE Distribution Files",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )
        section1.pack(fill=tk.X, pady=(0, 15))
        
        info_text = tk.Label(
            section1,
            text="Required files: convexity_kde.csv, roughness_kde.csv, eccentricity_kde.csv,\ncompactness_kde.csv, roundness_kde.csv",
            font=("Arial", 9),
            fg="#7F8C8D",
            justify=tk.LEFT
        )
        info_text.pack(anchor=tk.W, pady=(0, 10))
        
        self.kde_path_var = tk.StringVar()
        
        path_frame = tk.Frame(section1)
        path_frame.pack(fill=tk.X)
        
        self.kde_entry = tk.Entry(
            path_frame,
            textvariable=self.kde_path_var,
            font=("Arial", 10),
            state="readonly",
            width=50
        )
        self.kde_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(
            path_frame,
            text="Browse Folder",
            command=self.browse_kde_folder,
            bg="#3498DB",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        browse_btn.pack(side=tk.LEFT)
        
        # Section 2: Volume Distribution Parameters
        section2 = tk.LabelFrame(
            main_frame,
            text="Step 2: Volume Distribution (Gamma)",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )
        section2.pack(fill=tk.X, pady=(0, 15))
        
        params_frame = tk.Frame(section2)
        params_frame.pack(fill=tk.X)
        
        tk.Label(params_frame, text="Shape:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.shape_var = tk.DoubleVar(value=0.26)
        shape_entry = tk.Entry(params_frame, textvariable=self.shape_var, font=("Arial", 10), width=15)
        shape_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 30), pady=5)
        
        tk.Label(params_frame, text="Scale:", font=("Arial", 10)).grid(row=0, column=2, sticky=tk.W, pady=5)
        self.scale_var = tk.DoubleVar(value=1664.9)
        scale_entry = tk.Entry(params_frame, textvariable=self.scale_var, font=("Arial", 10), width=15)
        scale_entry.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Section 3: Generation Settings
        section3 = tk.LabelFrame(
            main_frame,
            text="Step 3: Generation Settings",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )
        section3.pack(fill=tk.X, pady=(0, 15))
        
        settings_frame = tk.Frame(section3)
        settings_frame.pack(fill=tk.X)
        
        tk.Label(settings_frame, text="Number of shapes:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_shapes_var = tk.IntVar(value=3)
        num_shapes_spin = tk.Spinbox(
            settings_frame,
            from_=1,
            to=20,
            textvariable=self.num_shapes_var,
            font=("Arial", 10),
            width=10
        )
        num_shapes_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 30), pady=5)
        
        tk.Label(settings_frame, text="Grid size:", font=("Arial", 10)).grid(row=0, column=2, sticky=tk.W, pady=5)
        self.grid_size_var = tk.IntVar(value=128)
        grid_size_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.grid_size_var,
            values=[64, 128, 256],
            font=("Arial", 10),
            width=10,
            state="readonly"
        )
        grid_size_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Section 4: Save Output
        section4 = tk.LabelFrame(
            main_frame,
            text="Step 4: Save Output (Optional)",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )
        section4.pack(fill=tk.X, pady=(0, 15))
        
        self.save_path_var = tk.StringVar()
        
        save_frame = tk.Frame(section4)
        save_frame.pack(fill=tk.X)
        
        self.save_entry = tk.Entry(
            save_frame,
            textvariable=self.save_path_var,
            font=("Arial", 10),
            state="readonly",
            width=50
        )
        self.save_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_save_btn = tk.Button(
            save_frame,
            text="Browse Folder",
            command=self.browse_save_folder,
            bg="#95A5A6",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        browse_save_btn.pack(side=tk.LEFT)
        
        # Generate Button
        generate_btn = tk.Button(
            main_frame,
            text="▶ Generate Shapes",
            command=self.generate_shapes,
            bg="#27AE60",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=15,
            cursor="hand2"
        )
        generate_btn.pack(pady=20)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please load KDE files to begin.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 9),
            bg="#ECF0F1",
            fg="#2C3E50",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def browse_kde_folder(self):
        """Open folder dialog to select KDE files folder."""
        folder = filedialog.askdirectory(
            title="Select folder containing KDE CSV files"
        )
        if folder:
            self.kde_folder = folder
            self.kde_path_var.set(folder)
            
            # Try to initialize generator
            try:
                self.generator = MorphologicalShapeGenerator(
                    kde_folder_path=folder,
                    volume_shape=self.shape_var.get(),
                    volume_scale=self.scale_var.get()
                )
                self.status_var.set(f"✓ Loaded KDE distributions from: {folder}")
                messagebox.showinfo(
                    "Success",
                    "KDE distributions loaded successfully!\n\n"
                    f"Folder: {folder}"
                )
            except Exception as e:
                self.status_var.set(f"✗ Error loading KDE files: {str(e)}")
                messagebox.showerror(
                    "Error",
                    f"Failed to load KDE distributions:\n\n{str(e)}"
                )
                self.generator = None
    
    def browse_save_folder(self):
        """Open folder dialog to select output folder."""
        folder = filedialog.askdirectory(
            title="Select folder to save generated shapes"
        )
        if folder:
            self.save_folder = folder
            self.save_path_var.set(folder)
            self.status_var.set(f"Output folder set: {folder}")
    
    def generate_shapes(self):
        """Generate shapes with current settings."""
        if self.generator is None:
            messagebox.showwarning(
                "No KDE Files",
                "Please load KDE distribution files first!"
            )
            return
        
        # Update generator parameters
        try:
            self.generator.volume_shape = self.shape_var.get()
            self.generator.volume_scale = self.scale_var.get()
        except:
            messagebox.showerror(
                "Invalid Parameters",
                "Please enter valid numeric values for shape and scale."
            )
            return
        
        num_shapes = self.num_shapes_var.get()
        grid_size = self.grid_size_var.get()
        
        # Confirm before starting
        save_info = f"\nSaving to: {self.save_folder}" if self.save_folder else "\nNot saving (visualization only)"
        
        response = messagebox.askyesno(
        "Confirm Generation",
        f"Generate {num_shapes} shapes with grid size {grid_size}x{grid_size}x{grid_size}?"
        f"{save_info}\n\n"
        "This may take several minutes. Continue?"
    )

        
        if not response:
            return
        
        # Disable UI during generation
        self.root.config(cursor="watch")
        self.status_var.set(f"Generating {num_shapes} shapes... Please wait.")
        self.root.update()
        
        try:
            # Generate shapes
            shapes, metrics = self.generator.generate_multiple_shapes(
                num_shapes=num_shapes,
                grid_size=grid_size,
                save_path=self.save_folder
            )
            
            # Show summary
            summary_text = f"Successfully generated {num_shapes} shapes!\n\n"
            summary_text += "Average metrics:\n"
            
            metrics_df = pd.DataFrame(metrics)
            for col in ['volume', 'eccentricity', 'roughness', 'convexity', 'compactness', 'roundness']:
                if col in metrics_df.columns:
                    avg = metrics_df[col].mean()
                    summary_text += f"  {col.capitalize()}: {avg:.3f}\n"
            
            if self.save_folder:
                summary_text += f"\n✓ Saved to: {self.save_folder}"
            
            messagebox.showinfo("Generation Complete", summary_text)
            self.status_var.set(f"✓ Generated {num_shapes} shapes successfully!")
            
        except Exception as e:
            messagebox.showerror(
                "Generation Error",
                f"An error occurred during generation:\n\n{str(e)}"
            )
            self.status_var.set(f"✗ Error: {str(e)}")
        
        finally:
            # Re-enable UI
            self.root.config(cursor="arrow")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = ShapeGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()