import subprocess
import os


workfolder = os.path.dirname(os.path.abspath(__file__))
main_cli_folder = os.path.abspath(os.path.join(workfolder, '..', '..','..'))
main_cli_path  = os.path.join(main_cli_folder,"main_cli.py")
# List of parameter files
parameter_files = [
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_90deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_60deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_45deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_30deg.ini"
]

# Run the command for each parameter file
for param_file in parameter_files:
    command = ["python3", main_cli_path, "-p", param_file]
    subprocess.run(command)
