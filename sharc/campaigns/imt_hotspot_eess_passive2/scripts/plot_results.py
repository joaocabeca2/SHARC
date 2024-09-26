import os
from sharc.plots.plot_cdf import plot_system_interference_power_from_imt_dl, plot_system_interference_power_from_imt_ul, plot_imt_station_antenna_gain_towards_system

# Define the base directory
name = "imt_hotspot_eess_passive2"

# this should behave similarly to `sharc/plots/plot_cdf:13`
# ideally the readable legend would be in the .ini metadata and all this code would be deleted
workfolder = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.abspath(os.path.join(workfolder, "..", "output"))

plot_system_interference_power_from_imt_dl(
    name,
    legends=None,
    subfolders=None,
    save_file=False,
    show_plot=True,
)

plot_system_interference_power_from_imt_ul(
    name,
    legends=None,
    subfolders=None,
    save_file=False,
    show_plot=True,
)
plot_imt_station_antenna_gain_towards_system(
    name,
    legends=None,
    subfolders=None,
    save_file=False,
    show_plot=True,
)


# Example with all subfolders and no specific legends
# This will include all subfolders that start with "output_imt_hibs_ras_2600_MHz_" in the base directory
# and generate legends automatically based on the folder names

# Define legends and subfolders as None to include all automatically
# legends = None
# subpastas = None

# Run the function with all subfolders and auto-generated legends
# plot_system_interference_power_from_imt_dl(name, legends=legends, subpastas=subpastas, save_file=True, show_plot=False)
