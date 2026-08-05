import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, poisson
from tkinter import Tk, filedialog
import os

# --------------------------
# Load CSV
# --------------------------
Tk().withdraw()
csv_file = filedialog.askopenfilename(title="Select stroke volume CSV",
                                      filetypes=[("CSV files","*.csv")])
if not csv_file:
    print("No file selected. Exiting.")
    exit()

df = pd.read_csv(csv_file)
volumes = df['Stroke_Volume'].values

# --------------------------
# Histogram
# --------------------------
plt.figure(figsize=(10,6))
counts, bins, _ = plt.hist(volumes, bins=50, alpha=0.6, color='gray', label='Stroke volumes')

# Bin centers for plotting PDF
bin_centers = (bins[:-1] + bins[1:]) / 2

# --------------------------
# Fit Gaussian
mu, std = norm.fit(volumes)
pdf_gauss = norm.pdf(bin_centers, mu, std) * len(volumes) * (bins[1]-bins[0])
plt.plot(bin_centers, pdf_gauss, 'r-', lw=2, label=f'Gaussian fit (μ={mu:.1f}, σ={std:.1f})')

# Fit Gamma
shape, loc, scale = gamma.fit(volumes, floc=0)
pdf_gamma = gamma.pdf(bin_centers, a=shape, loc=loc, scale=scale) * len(volumes) * (bins[1]-bins[0])
plt.plot(bin_centers, pdf_gamma, 'b--', lw=2, label=f'Gamma fit (shape={shape:.2f}, scale={scale:.1f})')

# Fit Poisson (only works for integers)
vol_int = np.round(volumes).astype(int)
lambda_poisson = np.mean(vol_int)
pdf_poisson = poisson.pmf(bin_centers.astype(int), lambda_poisson) * len(volumes)
plt.plot(bin_centers, pdf_poisson, 'g-.', lw=2, label=f'Poisson fit (λ={lambda_poisson:.1f})')

# --------------------------
# Plot settings
plt.xlabel("Stroke Volume (voxels)")
plt.ylabel("Count")
plt.title("Stroke Volume Distribution with Fitted Models")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_dir = os.path.dirname(csv_file)
output_file = os.path.join(output_dir, "stroke_volume_distribution_fits.png")
plt.savefig(output_file, dpi=300)
plt.show()
print(f"Saved plot: {output_file}")
