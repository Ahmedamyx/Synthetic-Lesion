import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.stats import gaussian_kde
import os

# --------------------------
# Helper Functions
# --------------------------

def fit_kde_and_save(data, descriptor_name, output_dir, bw_method='scott'):
    """Fit a kernel density estimator and save x/pdf to CSV."""
    data = data[~np.isnan(data)]
    if len(data) < 2:
        print(f"⚠️ Not enough data for {descriptor_name}. Skipping.")
        return None

    kde = gaussian_kde(data, bw_method=bw_method)
    
    x = np.linspace(min(data), max(data), 500)
    pdf_vals = kde(x)
    
    # Save x and pdf to CSV
    df_kde = pd.DataFrame({'x': x, 'pdf': pdf_vals})
    output_file = os.path.join(output_dir, f"{descriptor_name}_kde.csv")
    df_kde.to_csv(output_file, index=False)
    print(f"Saved KDE data to {output_file}")
    
    # Plot histogram + KDE
    plt.figure(figsize=(7,5))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', label='Data')
    plt.plot(x, pdf_vals, 'r-', lw=2, label='KDE fit')
    plt.title(f"{descriptor_name} distribution (KDE)")
    plt.xlabel(descriptor_name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{descriptor_name}_kde_fit.png"), dpi=300)
    plt.show()
    print(f"Saved plot: {descriptor_name}_kde_fit.png\n")
    
    return kde

# --------------------------
# Main Script
# --------------------------

if __name__ == "__main__":
    Tk().withdraw()
    print("Select the CSV file with stroke shape descriptors (excluding Volume)")
    csv_file = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not csv_file:
        print("No file selected. Exiting.")
        exit()

    output_dir = os.path.dirname(csv_file)
    
    df = pd.read_csv(csv_file)
    descriptors = ['Eccentricity', 'Compactness', 'Roundness', 'Convexity', 'Roughness']

    kde_models = {}
    for desc in descriptors:
        if desc not in df.columns:
            print(f"⚠️ {desc} not found in CSV. Skipping.")
            continue

        data = df[desc].dropna()
        if len(data) == 0:
            print(f"⚠️ No valid data for {desc}. Skipping.")
            continue

        print(f"Fitting KDE for {desc}...")
        kde = fit_kde_and_save(data, desc, output_dir)
        if kde is not None:
            kde_models[desc] = kde

    print("✅ All KDE fits completed and saved.")
