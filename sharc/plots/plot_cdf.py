import os
import pandas as pd
import plotly.graph_objects as go
import re
from sharc.plot import Plot
import numpy as np

class PlotCDF:

    def __init__(self, name: str):
        
        self.name = name
        self.n_bins = 1000

        # Define the base directory dynamically based on user input
        self.workfolder = os.path.dirname(os.path.abspath(__file__))
        self.csv_folder = os.path.abspath(os.path.join(self.workfolder, '..', "campaigns", name, "output"))
        self.figs_folder = os.path.abspath(os.path.join(self.workfolder, '..', "campaigns", name, "output", "figs"))

        self.plot_configurations = {
            'system_imt_antenna_gain': {
                'x_label': 'Antenna gain [dBi]',
                'y_label': 'Probability of antenna gain < $X$',
                'title': '[SYS] CDF of system antenna gain towards IMT stations',
            },
            'imt_system_antenna_gain': {
                'x_label': 'Antenna gain [dBi]',
                'y_label': 'Probability of antenna gain < $X$',
                'title': '[IMT] CDF of IMT station antenna gain towards system',
            },
            'imt_system_path_loss': {
                'x_label': 'Path Loss [dB]',
                'y_label': 'Probability of path loss < $X$',
                'title': '[SYS] CDF of IMT to system path loss',
            },
            'imt_system_build_entry_loss': {
                'x_label': 'Building entry loss [dB]',
                'y_label': 'Probability of loss < $X$',
                'title': '[SYS] CDF of IMT to system building entry loss',
            },
            'imt_system_diffraction_loss': {
                'x_label': 'Building entry loss [dB]',
                'y_label': 'Probability of loss < $X$',
                'title': '[SYS] CDF of IMT to system diffraction loss',
            },
            'imt_bs_antenna_gain': {
                'x_label': 'Antenna gain [dBi]',
                'y_label': 'Probability of antenna gain < $X$',
                'title': '[IMT] CDF of BS antenna gain towards the UE',
            },
            'imt_ue_antenna_gain': {
                'x_label': 'Antenna gain [dBi]',
                'y_label': 'Probability of antenna gain < $X$',
                'title': '[IMT] CDF of UE antenna gain towards the BS',
            },
            'imt_ul_tx_power_density': {
                'x_label': 'Transmit power density [dBm/Hz]',
                'y_label': 'Probability of transmit power density < $X$',
                'title': '[IMT] CDF of UE transmit power density',
            },
            'imt_ul_tx_power': {
                'x_label': 'Transmit power [dBm]',
                'y_label': 'Probability of transmit power < $X$',
                'title': '[IMT] CDF of UE transmit power',
            },
            'imt_ul_sinr_ext': {
                'x_label': 'SINR [dB]',
                'y_label': 'Probability of SINR < $X$',
                'title': '[IMT] CDF of UL SINR with external interference',
            },
            'imt_ul_sinr': {
                'x_label': 'SINR [dB]',
                'y_label': 'Probability of SINR < $X$',
                'title': '[IMT] CDF of UL SINR',
            },
            'imt_ul_snr': {
                'x_label': 'SNR [dB]',
                'y_label': 'Probability of SNR < $X$',
                'title': '[IMT] CDF of UL SNR',
            },
            'imt_ul_inr': {
                'x_label': '$I/N$ [dB]',
                'y_label': 'Probability of $I/N$ < $X$',
                'title': '[IMT] CDF of UL interference-to-noise ratio',
            },
            'imt_ul_tput_ext': {
                'x_label': 'Throughput [bits/s/Hz]',
                'y_label': 'Probability of UL throughput < $X$',
                'title': '[IMT] CDF of UL throughput with external interference',
            },
            'imt_ul_tput': {
                'x_label': 'Throughput [bits/s/Hz]',
                'y_label': 'Probability of UL throughput < $X$',
                'title': '[IMT] CDF of UL throughput',
            },
            'imt_path_loss': {
                'x_label': 'Path loss [dB]',
                'y_label': 'Probability of path loss < $X$',
                'title': '[IMT] CDF of path loss',
            },
            'imt_coupling_loss': {
                'x_label': 'Coupling loss [dB]',
                'y_label': 'Probability of coupling loss < $X$',
                'title': '[IMT] CDF of coupling loss',
            },
            'imt_dl_tx_power': {
                'x_label': 'Transmit power [dBm]',
                'y_label': 'Probability of transmit power < $X$',
                'title': '[IMT] CDF of DL transmit power',
            },
            'imt_dl_sinr_ext': {
                'x_label': 'SINR [dB]',
                'y_label': 'Probability of SINR < $X$',
                'title': '[IMT] CDF of DL SINR with external interference',
            },
            'imt_dl_sinr': {
                'x_label': 'SINR [dB]',
                'y_label': 'Probability of SINR < $X$',
                'title': '[IMT] CDF of DL SINR',
            },
            'imt_dl_snr': {
                'x_label': 'SNR [dB]',
                'y_label': 'Probability of SNR < $X$',
                'title': '[IMT] CDF of DL SNR',
            },
            'imt_dl_inr': {
                'x_label': '$I/N$ [dB]',
                'y_label': 'Probability of $I/N$ < $X$',
                'title': '[IMT] CDF of DL interference-to-noise ratio',
            },
            'imt_dl_tput_ext': {
                'x_label': 'Throughput [bits/s/Hz]',
                'y_label': 'Probability of throughput < $X$',
                'title': '[IMT] CDF of DL throughput with external interference',
            },
            'imt_dl_tput': {
                'x_label': 'Throughput [bits/s/Hz]',
                'y_label': 'Probability of throughput < $X$',
                'title': '[IMT] CDF of DL throughput',
            },
            'system_inr': {
                'x_label': 'INR [dB]',
                'y_label': 'Probability of INR < $X$',
                'title': '[SYS] CDF of system INR',
            },
            'system_pfd': {
                'x_label': 'PFD [dBm/m^2]',
                'y_label': 'Probability of INR < $X$',
                'title': '[SYS] CDF of system PFD',
            },
            'system_ul_interf_power': {
                'x_label': 'Interference Power [dBm]',
                'y_label': 'Probability of Power < $X$',
                'title': '[SYS] CDF of system interference power from IMT UL',
            },
            'system_dl_interf_power': {
                'x_label': 'Interference Power [dBm/MHz]',
                'y_label': 'Probability of Power < $X$',
                'title': '[SYS] CDF of system interference power from IMT DL',
            }
        }


    def plot_title_to_filename(self, title: str):
        """
        Creates the file name from the graph titles by removing spaces and brackets.
        """
        return re.sub(r'[\[\]]', "", title).replace(" ", "_")
    
    def cdf(self, data, file_name):
        """
        Compute the CDF of a given data.
        """
        values, base = np.histogram(data, bins=self.n_bins)
        cumulative = np.cumsum(values)
        x = base[:-1]
        y = cumulative / cumulative[-1]
        x_label = self.plot_configurations[file_name]['x_label']
        y_label = self.plot_configurations[file_name]['y_label']
        title = self.plot_configurations[file_name]['title']
        f_name = self.plot_title_to_filename(title)
        return Plot(x, y, x_label, y_label, title, f_name)
    
    def plot_cdf(self, variable, save_file=False, show_plot=True, subfolders=None):
        """
        Compute the CDF of a given simulation variable.
        """
        
        label = 'CDF'

        # Ensure at least one of save_file or show_plot is true
        if not save_file and not show_plot:
            raise ValueError("Either save_file or show_plot must be True.")
        
        # Check if the figs output folder exists, if not, create it
        if not os.path.exists(self.figs_folder):
            os.makedirs(self.figs_folder)

        # List all subfolders in the base directory or only those specified by the user
        if subfolders:
            subdirs = [os.path.join(self.csv_folder, d) for d in subfolders if os.path.isdir(os.path.join(self.csv_folder, d))]
        else:
            subdirs = [os.path.join(self.csv_folder, d) for d in os.listdir(self.csv_folder) 
                    if os.path.isdir(os.path.join(self.csv_folder, d)) and d.startswith(f"output_{self.name}_")]

        # Initialize global min and max values  
        global_min = float('inf')
        global_max = float('-inf')

        # First, calculate the global min and max values
        for subdir in subdirs:
            subdir = os.path.join(subdir, 'raw_data_dir')
            all_files = [f for f in os.listdir(subdir) if f.endswith('.csv')]

            for file_name in all_files:
                file_path = os.path.join(subdir, file_name)
                if os.path.exists(file_path):
                    try:
                        # Try reading the .csv file using pandas with different delimiters
                        try:
                            data = pd.read_csv(file_path, delimiter=',')
                        except:
                            data = pd.read_csv(file_path, delimiter=';')
                        
                        # Plot the CDF
                        plot_data = self.cdf(data, file_name)
                        data = pd.DataFrame({'x': plot_data.x, 'y': plot_data.y})

                        # Remove rows that do not contain valid numeric values
                        data = data.apply(pd.to_numeric, errors='coerce').dropna()
                        
                        if not data.empty:
                            global_min = min(global_min, data.iloc[:, 0].min())
                            global_max = max(global_max, data.iloc[:, 0].max())
                    except Exception as e:
                        print(f"Error processing the file {file_name}: {e}")

        # If no valid data was found, set reasonable defaults for global_min and global_max
        if global_min == float('inf') or global_max == float('-inf'):
            global_min, global_max = 0, 1
        
        fig = go.Figure()
        for subdir in subdirs:
            subdir = os.path.join(subdir, 'raw_data_dir')
            all_files = [f for f in os.listdir(subdir) if f.endswith('.csv')]

            for file_name in all_files:
                file_path = os.path.join(subdir, file_name)
                if os.path.exists(file_path):
                    try:
                        fig.add_trace(go.Scatter(x=plot_data.x, y=plot_data.y, mode='lines', name=plot_data.title))
                    except Exception as e:
                        print(f"Error processing the file {file_name}: {e}")

        # Graph configurations
        fig.update_layout(
            title='CDF Plot',
            xaxis_title='Value',
            yaxis_title='CDF',
            yaxis=dict(tickmode='array', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            xaxis=dict(tickmode='linear', tick0=global_min, dtick=5),
            legend_title="Labels"
        )

        # Show the figure if requested
        if show_plot:
            fig.show()
        
        # # Save the figure if requested
        # if save_file:
        #     fig_file_path = os.path.join(figs_folder, f"CDF_Plot_{file_prefix}.png")
        #     fig.write_image(fig_file_path)
        #     print(f"Figure saved: {fig_file_path}")



# def plot_cdf(base_dir, file_prefix, passo_xticks=5, xaxis_title='Value', legends=None, subfolders=None, save_file=True, show_plot=False):
    
#     label='CDF'
#     # Ensure at least one of save_file or show_plot is true
#     if not save_file and not show_plot:
#         raise ValueError("Either save_file or show_plot must be True.")
    
#     # Define the base directory dynamically based on user input
#     workfolder = os.path.dirname(os.path.abspath(__file__))
#     csv_folder = os.path.abspath(os.path.join(workfolder, '..', "campaigns", base_dir, "output"))
#     figs_folder = os.path.abspath(os.path.join(workfolder, '..', "campaigns", base_dir, "output", "figs"))

#     # Check if the figs output folder exists, if not, create it
#     if not os.path.exists(figs_folder):
#         os.makedirs(figs_folder)
    
#     # List all subfolders in the base directory or only those specified by the user
#     if subfolders:
#         subdirs = [os.path.join(csv_folder, d) for d in subfolders if os.path.isdir(os.path.join(csv_folder, d))]
#     else:
#         subdirs = [os.path.join(csv_folder, d) for d in os.listdir(csv_folder) 
#                 if os.path.isdir(os.path.join(csv_folder, d)) and d.startswith(f"output_{base_dir}_")]

#     # Validate the number of legends
#     if legends and len(legends) != len(subdirs):
#         raise ValueError("The number of provided legends does not match the number of found subfolders.")

#     # Initialize global min and max values
#     global_min = float('inf')
#     global_max = float('-inf')

#     # First, calculate the global min and max values
#     for subdir in subdirs:
#         all_files = [f for f in os.listdir(subdir) if f.endswith('.csv') and label in f and file_prefix in f]

#         for file_name in all_files:
#             file_path = os.path.join(subdir, file_name)
#             if os.path.exists(file_path):
#                 try:
#                     # Try reading the .csv file using pandas with different delimiters
#                     try:
#                         data = pd.read_csv(file_path, delimiter=',', skiprows=1)
#                     except:
#                         data = pd.read_csv(file_path, delimiter=';', skiprows=1)
                    
#                     # Ensure the data has at least two columns
#                     if data.shape[1] < 2:
#                         print(f"The file {file_name} does not have enough columns to plot.")
#                         continue

#                     # Remove rows that do not contain valid numeric values
#                     data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
#                     if not data.empty:
#                         global_min = min(global_min, data.iloc[:, 0].min())
#                         global_max = max(global_max, data.iloc[:, 0].max())
#                 except Exception as e:
#                     print(f"Error processing the file {file_name}: {e}")

#     # If no valid data was found, set reasonable defaults for global_min and global_max
#     if global_min == float('inf') or global_max == float('-inf'):
#         global_min, global_max = 0, 1

#     # Plot the graphs adjusting the axes
#     fig = go.Figure()
#     for idx, subdir in enumerate(subdirs):
#         all_files = [f for f in os.listdir(subdir) if f.endswith('.csv') and label in f and file_prefix in f]
#         legenda = legends[idx] if legends else os.path.basename(subdir).split(f"output_{base_dir}_")[1]

#         for file_name in all_files:
#             file_path = os.path.join(subdir, file_name)
#             if os.path.exists(file_path):
#                 try:
#                     # Try reading the .csv file using pandas with different delimiters
#                     try:
#                         data = pd.read_csv(file_path, delimiter=',', skiprows=1)
#                     except:
#                         data = pd.read_csv(file_path, delimiter=';', skiprows=1)
                    
#                     # Ensure the data has at least two columns
#                     if data.shape[1] < 2:
#                         print(f"The file {file_name} does not have enough columns to plot.")
#                         continue

#                     # Remove rows that do not contain valid numeric values
#                     data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
#                     # Check if there are enough data points to plot
#                     if data.empty or data.shape[0] < 2:
#                         print(f"The file {file_name} does not have enough data to plot.")
#                         continue
                    
#                     # Plot the CDF
#                     fig.add_trace(go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode='lines', name=f'{legenda}'))
#                 except Exception as e:
#                     print(f"Error processing the file {file_name}: {e}")
    
#     # Graph configurations
#     fig.update_layout(
#         title=f'CDF Plot for {file_prefix}',
#         xaxis_title=xaxis_title,
#         yaxis_title='CDF',
#         yaxis=dict(tickmode='array', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         xaxis=dict(tickmode='linear', tick0=global_min, dtick=passo_xticks),
#         legend_title="Labels"
#     )
    
#     # Show the figure if requested
#     if show_plot:
#         fig.show()
    
#     # Save the figure if requested
#     if save_file:
#         fig_file_path = os.path.join(figs_folder, f"CDF_Plot_{file_prefix}.png")
#         fig.write_image(fig_file_path)
#         print(f"Figure saved: {fig_file_path}")

# # Specific functions for different metrics
# def plot_bs_antenna_gain_towards_the_ue(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_BS_antenna_gain_towards_the_UE', passo_xticks, xaxis_title='Antenna Gain (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_coupling_loss(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_coupling_loss', passo_xticks, xaxis_title='Coupling Loss (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_dl_sinr(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_DL_SINR', passo_xticks, xaxis_title='DL SINR (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_dl_snr(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_DL_SNR', passo_xticks, xaxis_title='DL SNR (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_dl_throughput(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_DL_throughput', passo_xticks, xaxis_title='DL Throughput (Mbps)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_dl_transmit_power(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_DL_transmit_power', passo_xticks, xaxis_title='DL Transmit Power (dBm)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_imt_station_antenna_gain_towards_system(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_IMT_station_antenna_gain_towards_system', passo_xticks, xaxis_title='Antenna Gain (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_path_loss(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_path_loss', passo_xticks, xaxis_title='Path Loss (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_ue_antenna_gain_towards_the_bs(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'IMT_CDF_of_UE_antenna_gain_towards_the_BS', passo_xticks, xaxis_title='Antenna Gain (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_imt_to_system_path_loss(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'SYS_CDF_of_IMT_to_system_path_loss', passo_xticks, xaxis_title='Path Loss (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_system_antenna_towards_imt_stations(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'SYS_CDF_of_system_antenna_gain_towards_IMT_stations', passo_xticks, xaxis_title='Antenna Gain (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_system_inr(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'SYS_CDF_of_system_INR', passo_xticks, xaxis_title='INR (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_system_interference_power_from_imt_dl(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'SYS_CDF_of_system_interference_power_from_IMT_DL', passo_xticks, xaxis_title='Interference Power (dBm/MHz)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_system_pfd(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'SYS_CDF_of_system_PFD', passo_xticks, xaxis_title='PFD (dBW/m²)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# def plot_inr_samples(base_dir, passo_xticks=5, legends=None, subfolders=None, save_file=True, show_plot=True):
#     plot_cdf(base_dir, 'INR_samples', passo_xticks, xaxis_title='INR Samples (dB)', legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

# # Main function to identify labels and call the appropriate functions
# def all_plots(base_dir, legends=None, subfolders=None, save_file=True, show_plot=False):
#     plot_bs_antenna_gain_towards_the_ue(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_coupling_loss(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_dl_sinr(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_dl_snr(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_dl_throughput(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_dl_transmit_power(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_imt_station_antenna_gain_towards_system(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_path_loss(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_ue_antenna_gain_towards_the_bs(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_imt_to_system_path_loss(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_system_antenna_towards_imt_stations(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_system_inr(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_system_interference_power_from_imt_dl(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)
#     plot_system_pfd(base_dir, legends=legends, subfolders=subfolders, save_file=save_file, show_plot=show_plot)

if __name__ == "__main__":
    # Example usage
    name = "imt_hibs_ras_2600_MHz"
    plot = PlotCDF(name)
    plot.plot_cdf('system_imt_antenna_gain')


