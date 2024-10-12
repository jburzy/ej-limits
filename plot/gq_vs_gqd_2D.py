import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from atlasify import atlasify
from atlasify import monkeypatch_axis_labels

# Function to load data from the HDF5 file and structure it
def load_hdf5_data(file_path="limits.h5"):
    data = {}
    with h5py.File(file_path, 'r') as f:
        # Iterate through each group and store relevant data
        for group_name in f.keys():
            group = f[group_name]
            z_prime_mass = group['z_prime_mass'][()]
            piD_mass = group['piD_mass'][()]
            lifetime = group['lifetime'][()]
            limits = group['limits'][()]
            
            # Use a tuple of (z_prime_mass, pD_mass, lifetime) as the key
            data[(z_prime_mass, piD_mass, lifetime)] = limits
    return data

# Load the CSV data
df = pd.read_csv('results_summary.csv')

# Drop rows where 'cross_section' is NaN
df = df.dropna(subset=['cross_section'])

limits_data = load_hdf5_data("limits.h5")

for mass in [600, 1500, 3000]:

    # Filter the dataframe to only include entries where 'mass' equals 1500
    df_tmp = df[df['mass'] == mass]

    # Create a pivot table for heatmap
    heatmap_data = df_tmp.pivot_table(index='gqd_scaled', columns='gq_scaled', values='cross_section')
    heatmap_data = heatmap_data.sort_values(by='gqd_scaled', ascending=False)

    # Prepare data for interpolation using RegularGridInterpolator
    x = heatmap_data.columns.values  # gq_scaled
    y = heatmap_data.index.values    # gqd_scaled
    z = heatmap_data.values          # cross_section

    # Create an interpolator object using cubic interpolation
    interpolator = RegularGridInterpolator((y, x), z, method='linear', bounds_error=False, fill_value=None)

    # Create a dense grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the z values (cross section) on the dense grid
    zi = interpolator((yi, xi))

    # Check for NaN values in zi and handle them
    if np.any(np.isnan(zi)):
        print("NaN values found in the interpolated data. Filling NaNs with 0.")
        zi = np.nan_to_num(zi)  # Replace NaNs with 0

    # Plotting
    plt.figure(figsize=(8, 6))

    # Use LogNorm for logarithmic scaling for the heatmap
    heatmap = plt.imshow(zi, interpolation='nearest', cmap='viridis', aspect='auto',
                         extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', norm=LogNorm())

    try:
        # Draw the isotherm at z=100
        contour = plt.contour(xi, yi, zi, levels=[limits_data[(mass,20,5)][3]], colors='red', linewidths=2)

        # Shade the area **outside** the isotherm (for values above 100)
        plt.contourf(xi, yi, zi, levels=[limits_data[(mass,20,5)][3], zi.max()], colors='red', alpha=0.3)
    except Exception as e:
        print(f"Skipping contour drawing due to error: {e}")

    # Add colorbar and labels
    plt.colorbar(heatmap, label="$\sigma(pp \\rightarrow Z') \\times BR(Z' \\rightarrow q_D q_D)$ [fb]")
    monkeypatch_axis_labels()
    plt.xlabel('$g_{q}$')
    plt.ylabel('$g_{q_D}$')
    plt.grid(False)

    # Show plot
    atlasify("Internal", f"$\sqrt{{s}} = 13.6\,\mathrm{{TeV}}$, $m_{{Z'}} = {mass}\, \mathrm{{GeV}}, m_{{\pi_D}} = 20\,\mathrm{{GeV}}, c\\tau_{{\pi_D}}=5\, \mathrm{{mm}}$",outside=True)
    plt.savefig(f"gq_vs_gqd_2D_{mass}.pdf")
