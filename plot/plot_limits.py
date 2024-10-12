import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_limits_from_h5(filename="limits.h5"):
    """Read limits from an HDF5 file and return a dictionary of limits."""
    limits_data = {}
    with h5py.File(filename, "r") as h5file:
        limit_group = h5file["limits"]
        for dataset_name in limit_group:
            limits_data[dataset_name] = limit_group[dataset_name][:]
    return limits_data

def plot_limits(limits_data):
    """Plot the limits from the HDF5 file."""
    observed_limits = []
    expected_limits = []
    expected_m1sigma = []
    expected_m2sigma = []
    expected_p1sigma = []
    expected_p2sigma = []
    labels = []

    # Extract the limits from the data
    for label, limits in limits_data.items():
        labels.append(label)
        observed_limits.append(limits[0])
        expected_limits.append(limits[3])
        expected_m1sigma.append(limits[2])
        expected_m2sigma.append(limits[1])
        expected_p1sigma.append(limits[4])
        expected_p2sigma.append(limits[5])

    num_files = len(labels)
    width = 1.0 / num_files  # Width of each band
    x = np.arange(num_files) * width + width / 2  # Center the x positions

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.set_ylim(1e-2, 300)

    for i in range(num_files):
        x_pos = x[i]

        # Plot observed limits
        ax.plot([x_pos - width / 2, x_pos + width / 2], [observed_limits[i], observed_limits[i]], color='black', linewidth=2, label="Observed" if i == 0 else "")

        # Plot expected limits
        ax.plot([x_pos - width / 2, x_pos + width / 2], [expected_limits[i], expected_limits[i]], color='black', linestyle='--', linewidth=2, label="Expected" if i == 0 else "")

        # Plot the p2-m2 band
        ax.fill_betweenx([expected_m2sigma[i], expected_p2sigma[i]], x_pos - width / 2, x_pos + width / 2, color='yellow', alpha=0.5, label='Expected ±2σ' if i == 0 else "")

        # Plot the p1-m1 band
        ax.fill_betweenx([expected_m1sigma[i], expected_p1sigma[i]], x_pos - width / 2, x_pos + width / 2, color='green', alpha=0.5, label='Expected ±1σ' if i == 0 else "")

    # Add labels and title
    ax.set_ylabel('95% CL limit on σ [fb]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("limits_plot.pdf", format='pdf')

# Read limits from the HDF5 file and plot
limits_data = read_limits_from_h5("limits.h5")
plot_limits(limits_data)

