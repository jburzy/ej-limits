import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.colors import LogNorm
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

# We will store the (mass, gq) points where gqd=0.05 intersects the exclusion limit
intersection_points = []

# Loop over different masses
for mass in [600, 1500]:
    # Filter the dataframe to only include entries where 'mass' equals the current mass
    df_tmp = df[df['mass'] == mass]

    # Create a pivot table for heatmap
    heatmap_data = df_tmp.pivot_table(index='gqd_scaled', columns='gq_scaled', values='cross_section')

    # Check if gqd=0.05 is in the data
    if 0.1 in heatmap_data.index:
        gq_values = heatmap_data.columns.values  # gq_scaled
        cross_sections = heatmap_data.loc[0.1].values  # cross section at gqd=0.05

        # Find the exclusion limit for the current mass
        exclusion_limit = limits_data[(mass, 20, 5)][3]

        # Define a function to find the crossing point
        def crossing_function(gq):
            return np.interp(gq, gq_values, cross_sections) - exclusion_limit

        # Use fsolve to find the crossing point
        initial_guess = 0.01 if mass == 600 else 0.001# Starting point for the search
        crossing_point = fsolve(crossing_function, initial_guess)

        # Append valid crossing points
        if crossing_point[0] >= 0:  # Only consider positive gq values
            intersection_points.append((mass, crossing_point[0]))

# Convert intersection points to numpy array for easier handling
intersection_points = np.array(intersection_points)

# Plot the intersection points
plt.figure(figsize=(8, 6))
plt.plot(intersection_points[:, 0], intersection_points[:, 1], marker='o', linestyle='-', color='blue')
plt.xlabel("$m_{Z'} [GeV]$")
plt.ylabel('95% CL upper limit on $g_q$')

# Show the plot with Atlas style
atlasify("Internal", f"$\sqrt{{s}} = 13.6\,\mathrm{{TeV}}$, $m_{{\pi_D}} = 20\,\mathrm{{GeV}}, c\\tau_{{\pi_D}}=5\, \mathrm{{mm}}$, $g_{{q_D}} = 0.1$",outside=True)
plt.savefig("intersection_gqd_vs_mass.pdf")

