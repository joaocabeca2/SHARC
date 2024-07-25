import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# Caminho para a pasta de trabalho
workfolder = os.path.dirname(os.path.abspath(__file__))
main_cli_folder = os.path.abspath(os.path.join(workfolder, '..', '..', '..'))
main_cli_path  = os.path.join(main_cli_folder, "main_cli.py")

# Lista de arquivos de parâmetros
parameter_files = [
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_90deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_60deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_45deg.ini",
    "./campaigns/imt_mss_ras_2600_MHz_elevation/input/parameters_hibs_ras_2600_MHz_30deg.ini"
]

def run_command(param_file):
    command = ["python3", main_cli_path, "-p", param_file]
    subprocess.run(command)

# Número de threads (você pode ajustar conforme necessário)
num_threads = len(parameter_files)

# Executa os comandos em paralelo
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(run_command, parameter_files)
