import uproot
import numpy as np
import matplotlib.pyplot as plt
import pyhf
import h5py  # Import h5py to work with HDF5 files

# Define a simple class to hold file information
class FileData:
    def __init__(self, path, hist_name, label, cross_section):
        self.path = path
        self.hist_name = hist_name
        self.label = label
        self.cross_section = cross_section

def get_signal_yield_from_histogram(file_path, hist_name):
    """Opens a ROOT file using uproot and extracts the signal yield from a given histogram."""
    with uproot.open(file_path) as file:
        hist = file[hist_name]
        # get the N_EJ = 2 bin
        signal_yield = hist.values()[2]
    return signal_yield

def compute_limits(signal_yield, background_yield, cross_section):
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[signal_yield], bkg=[background_yield], bkg_uncertainty=[background_yield*0.1]
    )

    data = [background_yield] + model.config.auxdata
    poi_values = None

    if cross_section == 5000:
        poi_values = np.linspace(0.0, 0.05, 50)
    if cross_section == 100:
        poi_values = np.linspace(0.0, 0.01, 50)
    if cross_section == 2:
        poi_values = np.linspace(0.0, 1, 50)

    results = [
        pyhf.infer.hypotest(
            poi_value,
            data,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        for poi_value in poi_values
    ]

    observed = np.asarray([h[0] for h in results]).ravel()
    expected_m2sigma = np.asarray([h[1][0] for h in results]).ravel()
    expected_m1sigma = np.asarray([h[1][1] for h in results]).ravel()
    expected = np.asarray([h[1][2] for h in results]).ravel()
    expected_p1sigma = np.asarray([h[1][3] for h in results]).ravel()
    expected_p2sigma = np.asarray([h[1][4] for h in results]).ravel()

    obs_limit = np.interp(0.05, observed[::-1], poi_values[::-1]) * cross_section
    exp_limit_m2sigma = np.interp(0.05, expected_m2sigma[::-1], poi_values[::-1]) * cross_section
    exp_limit_m1sigma = np.interp(0.05, expected_m1sigma[::-1], poi_values[::-1]) * cross_section
    exp_limit = np.interp(0.05, expected[::-1], poi_values[::-1]) * cross_section
    exp_limit_p1sigma = np.interp(0.05, expected_p1sigma[::-1], poi_values[::-1]) * cross_section
    exp_limit_p2sigma = np.interp(0.05, expected_p2sigma[::-1], poi_values[::-1]) * cross_section

    return obs_limit, exp_limit_m2sigma, exp_limit_m1sigma, exp_limit, exp_limit_p1sigma, exp_limit_p2sigma

def compute_limits_from_root(file_data):
    signal_yield = get_signal_yield_from_histogram(file_data.path, file_data.hist_name)
    background_yield = 0
    if "Zp600" in file_data.path:
        background_yield = 111.9
    else:
        background_yield = 2

    return compute_limits(signal_yield, background_yield, file_data.cross_section)

def write_limits_to_h5(file_data_list, filename="limits.h5"):
    """Compute and write limits to an HDF5 file."""
    with h5py.File(filename, "w") as h5file:
        # Create a group for storing the limits
        limit_group = h5file.create_group("limits")

        for file_data in file_data_list:
            # Compute limits for each signal file
            obs_limit, exp_limit_m2, exp_limit_m1, exp_limit, exp_limit_p1, exp_limit_p2 = compute_limits_from_root(file_data)

            # Create a dataset for each file's limits, using the file label as the dataset name
            dataset_name = file_data.label.replace("\n", " ")  # Replace newlines with spaces for dataset names
            limits_data = [obs_limit, exp_limit_m2, exp_limit_m1, exp_limit, exp_limit_p1, exp_limit_p2]
            
            # Write the limits to the dataset under the file label group
            limit_group.create_dataset(dataset_name, data=limits_data)

    print(f"Limits successfully written to {filename}")

def plot_limits_for_multiple_files(file_data_list):
    observed_limits = []
    expected_limits = []
    expected_m1sigma = []
    expected_m2sigma = []
    expected_p1sigma = []
    expected_p2sigma = []

    # Loop through each file data, extract signal yield, and compute limits
    for file_data in file_data_list:
        obs_limit, exp_limit_m2, exp_limit_m1, exp_limit, exp_limit_p1, exp_limit_p2 = compute_limits_from_root(file_data)
        observed_limits.append(obs_limit)
        expected_limits.append(exp_limit)
        expected_m1sigma.append(exp_limit_m1)
        expected_m2sigma.append(exp_limit_m2)
        expected_p1sigma.append(exp_limit_p1)
        expected_p2sigma.append(exp_limit_p2)

    # Create a figure for the plot
    num_files = len(file_data_list)
    width = 1.0 / num_files  # Width of each band based on number of files
    x = np.arange(num_files) * width + width / 2  # Center the x positions

    fig, ax = plt.subplots()

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 300)  # Set limits between 0 and 1

    # Plot the p2-m2 band as a filled rectangle for each file
    for i in range(num_files):
        # Adjust the x position for spacing
        x_pos = x[i]

        # Plot observed limits as a solid black line
        ax.plot(
            [x_pos - width / 2, x_pos + width / 2],
            [observed_limits[i], observed_limits[i]],
            color='black', linewidth=2, label="Observed" if i == 0 else ""
        )

        # Plot expected limits as a dashed black line
        ax.plot(
            [x_pos - width / 2, x_pos + width / 2],
            [expected_limits[i], expected_limits[i]],
            color='black', linestyle='--', linewidth=2, label="Expected" if i == 0 else ""
        )

        # Plot the p2-m2 band
        ax.fill_betweenx(
            [expected_m2sigma[i], expected_p2sigma[i]],
            x_pos - width / 2, x_pos + width / 2,  # Width based on number of files
            color='yellow', alpha=0.5, label='Expected ±2σ' if i == 0 else ""
        )

        # Plot the p1-m1 band
        ax.fill_betweenx(
            [expected_m1sigma[i], expected_p1sigma[i]],
            x_pos - width / 2, x_pos + width / 2,  # Width based on number of files
            color='green', alpha=0.5, label='Expected ±1σ' if i == 0 else ""
        )

    # Add labels and title
    ax.set_ylabel('95% CL limit on σ [fb]')
    ax.set_xticks(x)  # Centered x-tick positions
    ax.set_xticklabels([file_data.label for file_data in file_data_list], fontsize=3)
    ax.legend()

    plt.xticks()
    plt.tight_layout()
    plt.savefig("limits.pdf", format='pdf')

# Example usage:
file_data_list = [
    FileData("Ld40_rho80_pi20_Zp600_l5.root", "EJ/n_emerging_jet", "$m_{Z'} = 0.6$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=5$ mm", 5000),
    FileData("Ld40_rho80_pi20_Zp600_l50.root", "EJ/n_emerging_jet", "$m_{Z'} = 0.6$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=50$ mm", 5000),
    FileData("Ld10_rho20_pi5_Zp600_l5.root", "EJ/n_emerging_jet", "$m_{Z'} = 0.6$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=5$ mm", 5000),
    FileData("Ld10_rho20_pi5_Zp600_l50.root", "EJ/n_emerging_jet", "$m_{Z'} = 0.6$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=50$ mm", 5000),
    FileData("Ld40_rho80_pi20_Zp1500_l5.root", "LargeR/n_emerging_jet", "$m_{Z'} = 1.5$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=5$ mm", 100),
    FileData("Ld40_rho80_pi20_Zp1500_l50.root", "LargeR/n_emerging_jet", "$m_{Z'} = 1.5$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=50$ mm", 100),
    FileData("Ld10_rho20_pi5_Zp1500_l5.root", "LargeR/n_emerging_jet", "$m_{Z'} = 1.5$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=5$ mm", 100),
    FileData("Ld10_rho20_pi5_Zp1500_l50.root", "LargeR/n_emerging_jet", "$m_{Z'} = 1.5$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=50$ mm", 100),
    FileData("Ld40_rho80_pi20_Zp3000_l5.root", "LargeR/n_emerging_jet", "$m_{Z'} = 3$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=5$ mm", 2),
    FileData("Ld40_rho80_pi20_Zp3000_l50.root", "LargeR/n_emerging_jet", "$m_{Z'} = 3$ TeV\n $m_{\pi_D} = 20$ GeV\n $c\\tau_{\pi_D}=50$ mm", 2),
    FileData("Ld10_rho20_pi5_Zp3000_l5.root", "LargeR/n_emerging_jet", "$m_{Z'} = 3$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=5$ mm", 2),
    FileData("Ld10_rho20_pi5_Zp3000_l50.root", "LargeR/n_emerging_jet", "$m_{Z'} = 3$ TeV\n $m_{\pi_D} = 5$ GeV\n $c\\tau_{\pi_D}=50$ mm", 2),
]

# Save the limits to an HDF5 file
write_limits_to_h5(file_data_list)

# Plot the limits
plot_limits_for_multiple_files(file_data_list)

