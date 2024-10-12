import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import h5py
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

# Drop rows where 'cross_section' or 'mass' is NaN
df = df.dropna(subset=['cross_section', 'mass'])

# Get unique values of mass
mass_values = sorted(df['mass'].unique())

# Prepare for plotting
plt.figure(figsize=(8, 6))

# Iterate over unique combinations of 'gq_scaled' and 'gqd_scaled'
for (gq_scaled, gqd_scaled), group in df.groupby(['gq_scaled', 'gqd_scaled']):
    if gq_scaled not in [0.005, 0.01]:
        continue
    if gqd_scaled not in [0.05]:
        continue

    # Sort the group by mass to ensure the interpolation works properly
    group = group.sort_values(by='mass')

    # Extract mass and cross-section values
    mass = group['mass'].values
    cross_section = group['cross_section'].values

    # Log-transform the cross-section
    log_cross_section = np.log(cross_section)

    # Interpolate in log-space
    log_interpolator = interp1d(mass, log_cross_section, kind='linear', fill_value='extrapolate')

    # Create a finer grid of mass values for a smooth curve
    mass_fine = np.linspace(min(mass), max(mass), 500)
    log_cross_section_fine = log_interpolator(mass_fine)

    # Convert interpolated log-values back to the original scale
    cross_section_fine = np.exp(log_cross_section_fine)

    # Plot the interpolated curve
    plt.plot(mass_fine, cross_section_fine, label=f'$g_q={gq_scaled}$, $g_{{q_D}}={gqd_scaled}$')

"""Plot the limits from the HDF5 file."""
observed_limits = []
expected_limits = []
expected_m1sigma = []
expected_m2sigma = []
expected_p1sigma = []
expected_p2sigma = []
labels = []

limits_data = load_hdf5_data("limits.h5")
print(limits_data)
masses = [600, 1500, 3000]

for mass in masses:
    limits = limits_data[(mass,20,5)]
    observed_limits.append(limits[0])
    expected_limits.append(limits[3])
    expected_m1sigma.append(limits[2])
    expected_m2sigma.append(limits[1])
    expected_p1sigma.append(limits[4])
    expected_p2sigma.append(limits[5])

# Plot the yellow band between top and bottom
plt.fill_between(masses, expected_m2sigma, expected_p2sigma, color='yellow', alpha=0.3)

# Plot the green band between second top and second bottom
plt.fill_between(masses, expected_m1sigma, expected_p1sigma, color='green', alpha=0.3)

# Plot the black line connecting the median points
plt.plot(masses,expected_limits, color='black', linewidth=2)


# Add labels and title
monkeypatch_axis_labels()
plt.yscale('log')
plt.xlabel("$m_{Z'}$ [GeV]")
plt.ylabel("$\sigma(pp \\rightarrow Z') \\times \mathrm{BR}(Z' \\rightarrow q_D q_D)$ [fb]")
plt.legend()
plt.grid(False)

# Show plot
atlasify("Internal", r"$\sqrt{s} = 13.6\,\mathrm{TeV}$, $m_{\pi_D} = 20\, \mathrm{GeV}, c\tau_{\pi_D}=5\, \mathrm{mm}$",outside=True)
plt.savefig("sigma_vs_mZ_bands.pdf")
plt.show()

