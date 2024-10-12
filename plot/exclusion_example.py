import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
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

# Define parameters
fixed_gqd = 0.05  # The gqd value we are interested in
mass_to_plot = 1500  # The mass point to visualize

# Filter the dataframe for the specified mass
df_tmp = df[df['mass'] == mass_to_plot]

# Create a pivot table for heatmap
heatmap_data = df_tmp.pivot_table(index='gqd_scaled', columns='gq_scaled', values='cross_section')

# Check if gqd=0.05 is in the data
if fixed_gqd in heatmap_data.index:
    gq_values = heatmap_data.columns.values  # gq_scaled
    cross_sections = heatmap_data.loc[fixed_gqd].values  # cross section at gqd=0.05

    # Find the exclusion limit for the current mass
    exclusion_limit = limits_data[(mass_to_plot, 20, 5)][3]

    # Define the interpolation function for cross sections
    interp_func = RegularGridInterpolator((gq_values,), cross_sections, bounds_error=False, fill_value=None)

    # Create a dense grid for gq values for plotting
    gq_dense = np.linspace(gq_values.min(), gq_values.max(), 200)
    cross_section_interp = interp_func(gq_dense)

    # Define a function to find the crossing point
    def crossing_function(gq):
        return np.interp(gq, gq_values, cross_sections) - exclusion_limit

    # Use fsolve to find the crossing point
    initial_guess = 0.001  # Starting point for the search
    crossing_point = fsolve(crossing_function, initial_guess)

    # Plot the interpolated cross section
    plt.figure(figsize=(8, 6))
    plt.plot(gq_dense, cross_section_interp, label='Interpolated Cross Section', color='blue')
    plt.axhline(y=exclusion_limit, color='red', linestyle='--', label='Exclusion Limit')
    
    # Mark the intersection point if found
    if crossing_point[0] >= 0 and crossing_point[0] <= gq_values.max():
        crossing_y_value = np.interp(crossing_point[0], gq_values, cross_sections)
        plt.plot(crossing_point, crossing_y_value, 'ro', label='Intersection Point')

    # Plot the original data points
    plt.scatter(gq_values, cross_sections, color='green', label='Original Data Points', zorder=5)

    # Add labels and legend
    plt.xlabel('$g_q$')
    plt.ylabel("$\sigma(pp \\rightarrow Z') \\times \mathrm{BR}(Z' \\rightarrow q_D q_D)$ [fb]")
    plt.ylim(0, np.max(cross_section_interp) * 1.2)  # Adjust y-limits for better visibility
    plt.legend()

    # Show the plot with Atlas style
    atlasify("Internal", f"$\sqrt{{s}} = 13.6\,\mathrm{{TeV}}$, $m_{{\pi_D}} = 20\,\mathrm{{GeV}}, c\\tau_{{\pi_D}}=5\, \mathrm{{mm}}$",outside=True)
    plt.savefig(f"cross_section_vs_gq_mass_{mass_to_plot}.pdf")
else:
    print(f"gqd={fixed_gqd} not found in the data for mass {mass_to_plot}.")

