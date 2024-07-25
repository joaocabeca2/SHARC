import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# Path to the working directory
workfolder = os.path.dirname(os.path.abspath(__file__))
main_cli_folder = os.path.abspath(os.path.join(workfolder, '..', '..', '..'))
main_cli_path  = os.path.join(main_cli_folder, "main_cli.py")

# List of parameter files
parameter_files = [
    "./campaigns/imt_mss_ras_2600_MHz_elevation_fspl/input/parameters_hibs_ras_2600_MHz_fspl_90deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation_fspl/input/parameters_hibs_ras_2600_MHz_fspl_60deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation_fspl/input/parameters_hibs_ras_2600_MHz_fspl_45deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation_fspl/input/parameters_hibs_ras_2600_MHz_fspl_30deg.ini"
]

def run_command(param_file):
    command = ["python3", main_cli_path, "-p", param_file]
    subprocess.run(command)

# Number of threads
num_threads = len(parameter_files)

# Run the commands in parallel
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(run_command, parameter_files)
