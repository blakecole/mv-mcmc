# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: mv_mcmc_plot_h.py                                 #
#    DATE: 20 MAR 2025                                       #
# ********************************************************** #

# PURPOSE: Plot optimized heading sequences vs. time for each start time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os

# Set plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['figure.dpi'] = 600
plt.style.use('dark_background')

# Define timestep in hours and convert to minutes
dt_hours = 0.005
dt_minutes = dt_hours * 60  # 0.3 minutes

# Find all CSV files matching the pattern "mv_swim_heading_nnnn.csv"
pattern = "../out/mv_swim_heading_*.csv"
csv_files = glob.glob(pattern)

# Check if any files were found
if not csv_files:
    raise ValueError("No CSV files found matching the pattern: " +  pattern)


# Helper function to extract the starting time (hhmm) from the filename
def extract_start_time(file_path):
    # Filename format: "mv_swim_heading_nnnn.csv"
    base = os.path.basename(file_path)
    time_str = base.split("_")[-1].split(".")[0]  # Extract "nnnn"
    return int(time_str)

# Sort files based on the extracted starting time
csv_files = sorted(csv_files, key=extract_start_time)

# Extract all start times for normalization
start_times = [extract_start_time(file) for file in csv_files]

# Create a normalization instance based on the minimum and maximum start times
norm = mcolors.Normalize(vmin=min(start_times), vmax=max(start_times))

# Choose a colormap (e.g., 'viridis')
cmap = plt.get_cmap('plasma')

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Loop over each CSV file to load, process, and plot the data
for file in csv_files:
    # Extract the starting time and format it as a 4-digit string
    start_time = extract_start_time(file)
    start_time_str = f"{start_time:04d}"
    
    # Format as hh:mm by inserting a colon
    formatted_time = start_time_str[:2] + ':' + start_time_str[2:]
    
    # Get the corresponding color from the colormap
    color = cmap(norm(start_time))
    
    # Load the heading data from the CSV file
    headings = np.loadtxt(file, delimiter=',')
    
    # Generate a time axis in minutes
    time_axis = np.arange(len(headings)) * dt_minutes
    
    # Plot the headings versus time using the computed color and label
    plt.plot(time_axis, headings, color=color, label=f"{formatted_time}")

# Label the axes, add a title, and create a legend with a description
plt.xlabel("Time [min]")
plt.ylabel("Heading [deg]")
plt.title("Swim Heading vs. Time")
plt.legend(title="Start Time")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
