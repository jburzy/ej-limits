import uproot
import numpy as np
import h5py  # Import h5py to work with HDF5 files
import pyhf

class ModelProperties:
    def __init__(self, z_prime_mass, pD_mass, lifetime, label):
        self.z_prime_mass = z_prime_mass  # Z' mass in GeV
        self.pD_mass = pD_mass              # pD mass in GeV
        self.lifetime = lifetime              # Lifetime in mm
        self.label = label                    # Label in LaTeX

class FileData:
    def __init__(self, path, hist_name, particle_properties, cross_section):
        self.path = path
        self.hist_name = hist_name
        self.particle_properties = particle_properties
        self.cross_section = cross_section

def get_signal_yield_from_histogram(file_path, hist_name):
    """Opens a ROOT file using uproot and extracts the signal yield from a given histogram."""
    with uproot.open(file_path) as file:
        hist = file[hist_name]
        # Get the N_EJ = 2 bin
        signal_yield = hist.values()[2]
    return signal_yield

def compute_limits(signal_yield, background_yield, cross_section):
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[signal_yield], bkg=[background_yield], bkg_uncertainty=[background_yield * 0.1]
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
    """Compute and write limits to an HDF5 file, including particle properties."""
    with h5py.File(filename, "w") as h5file:
        # Create a group for storing the limits

        for file_data in file_data_list:
            # Compute limits for each signal file
            obs_limit, exp_limit_m2, exp_limit_m1, exp_limit, exp_limit_p1, exp_limit_p2 = compute_limits_from_root(file_data)

            # Create a dataset for each file's limits, using the file label as the dataset name
            dataset_name = file_data.particle_properties.label.replace("\n", " ")  # Replace newlines with spaces for dataset names
            limits_data = [obs_limit, exp_limit_m2, exp_limit_m1, exp_limit, exp_limit_p1, exp_limit_p2]

            limit_group = h5file.create_group(file_data.particle_properties.label)

            # Write the limits to the dataset under the file label group
            limit_group.create_dataset("limits", data=limits_data)
            limit_group.create_dataset("z_prime_mass", data=file_data.particle_properties.z_prime_mass)
            limit_group.create_dataset("piD_mass", data=file_data.particle_properties.pD_mass)
            limit_group.create_dataset("lifetime", data=file_data.particle_properties.lifetime)

    print(f"Limits and properties successfully written to {filename}")

# Example usage of ModelProperties and FileData
file_data_list = [
    FileData("Ld40_rho80_pi20_Zp600_l5.root", "EJ/n_emerging_jet",
              ModelProperties(z_prime_mass=600, pD_mass=20, lifetime=5, label="Ld40_rho80_pi20_Zp600_l5"),
              cross_section=5000),
    FileData("Ld40_rho80_pi20_Zp600_l50.root", "EJ/n_emerging_jet",
              ModelProperties(z_prime_mass=600, pD_mass=20, lifetime=50, label="Ld40_rho80_pi20_Zp600_l50"),
              cross_section=5000),
    FileData("Ld10_rho20_pi5_Zp600_l5.root", "EJ/n_emerging_jet",
              ModelProperties(z_prime_mass=600, pD_mass=5, lifetime=5, label="Ld10_rho20_pi5_Zp600_l5"),
              cross_section=5000),
    FileData("Ld10_rho20_pi5_Zp600_l50.root", "EJ/n_emerging_jet",
              ModelProperties(z_prime_mass=600, pD_mass=5, lifetime=50, label="Ld10_rho20_pi5_Zp600_l50"),
              cross_section=5000),
    FileData("Ld40_rho80_pi20_Zp1500_l5.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=1500, pD_mass=20, lifetime=5, label="Ld40_rho80_pi20_Zp1500_l5"),
              cross_section=100),
    FileData("Ld40_rho80_pi20_Zp1500_l50.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=1500, pD_mass=20, lifetime=50, label="Ld40_rho80_pi20_Zp1500_l50"),
              cross_section=100),
    FileData("Ld10_rho20_pi5_Zp1500_l5.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=1500, pD_mass=5, lifetime=5, label="Ld10_rho20_pi5_Zp1500_l5"),
              cross_section=100),
    FileData("Ld10_rho20_pi5_Zp1500_l50.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=1500, pD_mass=5, lifetime=50, label="Ld10_rho20_pi5_Zp1500_l50"),
              cross_section=100),
    FileData("Ld40_rho80_pi20_Zp3000_l5.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=3000, pD_mass=20, lifetime=5, label="Ld40_rho80_pi20_Zp3000_l5"),
              cross_section=2),
    FileData("Ld40_rho80_pi20_Zp3000_l50.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=3000, pD_mass=20, lifetime=50, label="Ld40_rho80_pi20_Zp3000_l50"),
              cross_section=2),
    FileData("Ld10_rho20_pi5_Zp3000_l5.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=3000, pD_mass=5, lifetime=5, label="Ld10_rho20_pi5_Zp150_l5"),
              cross_section=2),
    FileData("Ld10_rho20_pi5_Zp3000_l50.root", "LargeR/n_emerging_jet",
              ModelProperties(z_prime_mass=3000, pD_mass=5, lifetime=50, label="Ld10_rho20_pi5_Zp150_l50"),
              cross_section=2),
]

write_limits_to_h5(file_data_list)

