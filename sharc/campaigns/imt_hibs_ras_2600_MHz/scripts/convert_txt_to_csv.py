import os
import pandas as pd
import sys

def save_to_csv(label, label2):
    # Get the current directory of the script
    workfolder = os.path.dirname(os.path.abspath(__file__))
    
    # Define the base path dynamically based on the current directory
    name_campaigns = "imt_hibs_ras_2600_MHz"
    base_path = os.path.abspath(os.path.join(workfolder, '..', 'output'))
    subfolder_name = "output_" + name_campaigns + "_" + label
    input_folder = os.path.join(base_path, subfolder_name)
    
    # Check if the directory exists
    if not os.path.exists(input_folder):
        print(f"The directory {input_folder} does not exist.")
        return

    # Create the output/RAS folder if it does not exist
    output_ras_folder = os.path.join(base_path, 'RAS_distance_csv')
    if not os.path.exists(output_ras_folder):
        os.makedirs(output_ras_folder)

    # List all .txt files in the specified directory
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print("Found .txt files:", all_files)

    for file in all_files:
        try:
            # Read the .txt file using pandas, assuming there are two columns
            file_path = os.path.join(input_folder, file)
            data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[0, 1])
            
            # Create the name of the .csv file with the same name as the .txt file and add label2
            base_name = os.path.splitext(file)[0]
            csv_name = f"RAS_{label2}_{base_name}.csv"
            csv_path = os.path.join(output_ras_folder, csv_name)
            
            # Save the data to a .csv file
            data.to_csv(csv_path, index=False)
            print(f"File saved: {csv_path}")
        except Exception as e:
            print(f"Error processing the file {file}: {e}")

if __name__ == '__main__':
    dist = "500"
    label = dist + "km_2024-07-24_01"
    label2 = dist + "km"

    save_to_csv(label, label2)
