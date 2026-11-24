import subprocess
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor


def run_command(param_file, main_cli_path):
    """
    Run the main_cli.py script with the specified parameter file.

    Args:
        param_file (str): Path to the parameter file.
        main_cli_path (str): Path to the main_cli.py script.
    """
    command = [sys.executable, main_cli_path, "-p", param_file]
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"FAILED ({result.returncode}): {param_file}", file=sys.stderr)
    return result.returncode


def run_campaign(campaign_name):
    """
    Run a campaign by executing main_cli.py for each parameter file in the campaign's input directory using multiple threads.

    Args:
        campaign_name (str): Name of the campaign to run.
    """
    # Path to the working directory
    workfolder = os.path.dirname(os.path.abspath(__file__))
    main_cli_path = os.path.join(workfolder, "main_cli.py")

    # Campaign directory
    campaign_folder = os.path.join(
        workfolder, "campaigns", campaign_name, "input",
    )

    # List of parameter files
    parameter_files = [
        os.path.join(campaign_folder, f) for f in os.listdir(
            campaign_folder,
        ) if f.endswith('.yaml')
    ]

    if len(parameter_files) == 0:
        raise ValueError(
            f"No parameter files were found in {campaign_folder}"
        )

    # Number of threads (adjust as needed)
    num_threads = min(len(parameter_files), os.cpu_count())

    # Run the commands in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(
            run_command, parameter_files, [
                main_cli_path,
            ] * len(parameter_files),
        )


def run_campaign_re(campaign_name, param_name_regex):
    """
    Run a campaign for parameter files matching a given regular expression.

    Execute main_cli.py for each parameter file in the specified campaign's input directory
    whose filename matches the given regular expression.

    Args:
        campaign_name (str): Name of the campaign.
        param_name_regex (str): Regular expression to filter parameter file names.
    """
    # Path to the working directory
    workfolder = os.path.dirname(os.path.abspath(__file__))
    main_cli_path = os.path.join(workfolder, "main_cli.py")

    # Campaign directory
    campaign_folder = os.path.join(
        workfolder, "campaigns", campaign_name, "input",
    )

    # List of parameter files
    pat = re.compile(param_name_regex)
    parameter_files = [
        os.path.join(campaign_folder, f) for f in os.listdir(
            campaign_folder,
        ) if pat.match(f)
    ]

    # Number of threads (adjust as needed)
    num_threads = min(len(parameter_files), os.cpu_count())

    # Run the commands in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(
            run_command, parameter_files, [
                main_cli_path,
            ] * len(parameter_files),
        )


def wifi_case_cost(param_file):
    """
    Rough relative cost of a WIFI campaign case, from its file name
    (..._WIFI_yy_NA_zz.yaml): number of APs times Wi-Fi load.
    """
    m = re.search(r"_WIFI_(\d+)_NA_(\d+)\.yaml$", os.path.basename(param_file))
    return int(m.group(2)) * int(m.group(1)) if m else 0


def run_campaigns(campaign_names):
    """
    Run several campaigns in a single pool of processes, starting with the
    most expensive cases so the longest one does not end up last.

    Args:
        campaign_names (list): Names of the campaigns to run.
    """
    workfolder = os.path.dirname(os.path.abspath(__file__))
    main_cli_path = os.path.join(workfolder, "main_cli.py")

    parameter_files = []
    for campaign_name in campaign_names:
        campaign_folder = os.path.join(workfolder, "campaigns", campaign_name, "input")
        parameter_files += [
            os.path.join(campaign_folder, f)
            for f in os.listdir(campaign_folder) if f.endswith('.yaml')
        ]
    if len(parameter_files) == 0:
        raise ValueError(f"No parameter files were found in {campaign_names}")

    parameter_files.sort(key=wifi_case_cost, reverse=True)
    num_threads = min(len(parameter_files), os.cpu_count())

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        return_codes = list(executor.map(
            run_command, parameter_files, [main_cli_path] * len(parameter_files),
        ))
    failed = sum(rc != 0 for rc in return_codes)
    print(f"{len(parameter_files) - failed}/{len(parameter_files)} cases finished OK")


if __name__ == "__main__":
    # Example usage
    #campaign_name = sys.argv[1]
    run_campaign('IMT_WIFI_URBANO_LF_IMT_COCANAL')
