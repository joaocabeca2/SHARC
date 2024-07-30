
## Setup the Campaigns Folder

1. Create a Folder for Your Campaign:
   - Example: `imt_hibs_ras_2600_MHz`

2. Inside the Campaign Folder, Create the Following Subfolders:
   - `input` -> This folder is for defining the simulation parameters. For example, look in `campaigns/imt_hibs_ras_2600_MHz/input`.
   - `output` -> SHARC will use this folder to save the results and plots.
   - `scripts` -> This folder is used to implement scripts for post-processing.

## Configure Your Simulation

1. Create a Parameter File:
   - Example: `campaigns/imt_hibs_ras_2600_MHz/input/parameters_hibs_ras_2600_MHz_0km.ini`.

2. Set the Configuration for Your Study in the Parameter File.

3. Set the Output Folder in the Parameter File:
    ```ini
    ###########################################################################
    # Output destination folder - this is relative to the SHARC/sharc directory
    output_dir = campaigns/imt_hibs_ras_2600_MHz/output/
    ###########################################################################
    # Output folder prefix
    output_dir_prefix = output_imt_hibs_ras_2600_MHz_0km
    ```
4. You Can Create Multiple Simulation Parameters:
   - Check the folder `sharc/campaigns/imt_hibs_ras_2600_MHz/input` for examples.

## Run Simulations and View the Results

1. **Run Simulations:**
   - In the `scripts` folder, you can create a file to start the simulation. Check the example: `campaigns/imt_hibs_ras_2600_MHz/scripts/start_simulations_multi_thread.py`.

   - If you want to run a single-threaded simulation, check the file: `campaigns/imt_hibs_ras_2600_MHz/scripts/start_simulations_single_thread.py`.

2. **Generate Plots:**
   - You can create a file to read the data and generate the plots. SHARC has a function called `plot_cdf` to make plotting easy. Check the example: `campaigns/imt_hibs_ras_2600_MHz/scripts/plot_results.py`.
