import uproot
import numpy as np

# Define the paths to the signal and background ROOT files
signal_files = ["Ld10_rho20_pi5_Zp600_l5_EJ.root", "Ld10_rho20_pi5_Zp600_l50_EJ.root","Ld40_rho80_pi20_Zp600_l5_EJ.root", "Ld40_rho80_pi20_Zp600_l50_EJ.root"]  # Add paths to your signal files
background_file = "QCD_LR.root"  # Path to your background file

# Histogram path in the ROOT file
histogram_path = "CR1Tag/n_emerging_jet"

# Function to get the content of the 2nd bin from a TH1F histogram
def get_second_bin_content(file_path, hist_path):
    with uproot.open(file_path) as root_file:
        hist = root_file[hist_path]
        bin_contents = hist.values()  # Get bin contents as a NumPy array
        return bin_contents[1]  # Index 1 corresponds to the 2nd bin

# Get the 2nd bin content for the background
background_content = get_second_bin_content(background_file, histogram_path)
print(background_content)

# Initialize a dictionary to store S/B values for each signal
sb_ratios = {}

# Loop over each signal file and calculate S/B
for signal_file in signal_files:
    signal_content = get_second_bin_content(signal_file, histogram_path)
    print(signal_content)
    
    if background_content > 0:  # Avoid division by zero
        sb_ratio = signal_content / background_content
    else:
        sb_ratio = np.inf  # Set S/B to infinity if background content is zero
    
    sb_ratios[signal_file] = sb_ratio * 100

# Output the S/B ratios
for signal_file, sb_ratio in sb_ratios.items():
    print(f"S/B ratio for {signal_file}: {sb_ratio:.3f}")

