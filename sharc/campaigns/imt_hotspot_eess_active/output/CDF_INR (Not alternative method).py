import csv
import numpy as np
import matplotlib.pyplot as plt

# Define paths to your CSV files (replace with actual file paths)
csv_file_DL_SM = 'path_to_INR_DL_SM.csv'
csv_file_DL_LG = 'path_to_INR_DL_LG.csv'
csv_file_UL_SM = 'path_to_INR_UL_SM.csv'
csv_file_UL_LG = 'path_to_INR_UL_LG.csv'

def load_data_from_csv(file_path):
    """
    Load data from a CSV file, skipping the first two rows.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.array: Array containing the loaded data from the CSV.
    """
    try:
    
        """
        Load data from CSV, skipping the first two rows 
        (First row indicates how many snapshots are generated in simulation and the second row indicates x and y axis)
        """
    
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the first row (snapshots info)
            next(reader)  # Skip the second row (axis labels)
            data = [float(row[0]) for row in reader]
            return np.array(data)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return np.array([])

# Load data from the CSV files
INR_DL_SM = load_data_from_csv(csv_file_DL_SM)
INR_DL_LG = load_data_from_csv(csv_file_DL_LG)
INR_UL_SM = load_data_from_csv(csv_file_UL_SM)
INR_UL_LG = load_data_from_csv(csv_file_UL_LG)

# Ensure that all data arrays were successfully loaded before proceeding
if not all([INR_DL_SM.size, INR_DL_LG.size, INR_UL_SM.size, INR_UL_LG.size]):
    raise ValueError("Error: One or more datasets failed to load. Please check the CSV file paths.")

# Convert INR values from dB to linear scale
INR_DL_SM_lin = 10 ** (INR_DL_SM / 10)
INR_DL_LG_lin = 10 ** (INR_DL_LG / 10)
INR_UL_SM_lin = 10 ** (INR_UL_SM / 10)
INR_UL_LG_lin = 10 ** (INR_UL_LG / 10)

# Aggregate INR values for DL and UL in linear scale
INR_DL = INR_DL_SM_lin + INR_DL_LG_lin
INR_UL = INR_UL_SM_lin + INR_UL_LG_lin

# Generate random indices for aggregating different combinations of DL and UL
Ind1 = np.random.randint(0, len(INR_DL), len(INR_DL))
Ind2 = np.random.randint(0, len(INR_DL), len(INR_DL))
Ind3 = np.random.randint(0, len(INR_DL), len(INR_DL))
Ind4 = np.random.randint(0, len(INR_DL), len(INR_DL))
Ind5 = np.random.randint(0, len(INR_DL), len(INR_DL))
Ind6 = np.random.randint(0, len(INR_DL), len(INR_DL))

# Calculate aggregate INR values using a weighted combination of DL and UL
INR_Agg_1 = 10 * np.log10(0.75 * (INR_DL_SM_lin[Ind3] + INR_DL_LG_lin[Ind4]) + 0.25 * (INR_UL_SM_lin[Ind5] + INR_UL_LG_lin[Ind6]))
INR_Agg_2 = 10 * np.log10(0.75 * INR_DL[Ind1] + 0.25 * INR_UL[Ind2])

def cdf_empirical(data):
    """
    Calculate the empirical CDF (Cumulative Distribution Function) of the data.

    Args:
        data (np.array): Input data for which the CDF is to be calculated.

    Returns:
        np.array: Two-column array where the first column is sorted data and
                  the second column is the CDF values.
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return np.column_stack((sorted_data, cdf))

# Compute empirical CDFs for the loaded INR data
CDF_INR_DL_LG = cdf_empirical(INR_DL_LG)
CDF_INR_DL_SM = cdf_empirical(INR_DL_SM)
CDF_INR_UL_LG = cdf_empirical(INR_UL_LG)
CDF_INR_UL_SM = cdf_empirical(INR_UL_SM)
CDF_INR_Agg_1 = cdf_empirical(INR_Agg_1)
CDF_INR_Agg_2 = cdf_empirical(INR_Agg_2)

# Plotting the CDFs for different INR data sets
plt.figure()
plt.plot(CDF_INR_DL_LG[:, 0], CDF_INR_DL_LG[:, 1], linewidth=2, label='DL - large beam')
plt.plot(CDF_INR_UL_LG[:, 0], CDF_INR_UL_LG[:, 1], linewidth=2, label='UL - large beam')
plt.plot(CDF_INR_DL_SM[:, 0], CDF_INR_DL_SM[:, 1], linewidth=2, label='DL - 3dB spotbeam')
plt.plot(CDF_INR_UL_SM[:, 0], CDF_INR_UL_SM[:, 1], linewidth=2, label='UL - 3dB spotbeam')
plt.plot(CDF_INR_Agg_1[:, 0], CDF_INR_Agg_1[:, 1], linewidth=2, label='Aggregate CDF 1')
plt.plot(CDF_INR_Agg_2[:, 0], CDF_INR_Agg_2[:, 1], linewidth=2, label='Aggregate CDF 2')

# Customize plot labels, title, and grid
plt.legend()
plt.xlabel('Interference to Noise Ratio [dB]')
plt.ylabel('CDF [%]')
plt.title('CDF of Interference to Noise Ratio for 18° Nadir - EESS Active')
plt.grid(True)
plt.show()
