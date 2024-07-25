import os
import pandas as pd
import plotly.graph_objects as go

def plot_cdf(file_prefix, label='CDF', valores_label=['30', '45', '60'], passo_xticks=5, xaxis_title='Value'):
    # Get the current directory of the script
    workfolder = os.path.dirname(os.path.abspath(__file__))
    
    # Define the base path dynamically based on the current directory
    csv_folder = os.path.abspath(os.path.join(workfolder, '..', "output", "RAS_elevation_csv"))
    figs_folder = os.path.abspath(os.path.join(workfolder, '..', "output", "RAS_elevation_figs"))

    # Check if the csv folder exists, if not, create it
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    # Check if the figs folder exists, if not, create it
    #if not os.path.exists(figs_folder):
    #    os.makedirs(figs_folder)
    
    # Get all .csv files in the directory that contain the label in the name
    all_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv') and label in f and file_prefix in f]
    
    # Get the list of base file names without the label
    base_names = set(f.split('deg_')[1] for f in all_files if file_prefix in f)
    
    # Initialize global min and max values
    global_min = float('inf')
    global_max = float('-inf')

    # First, calculate the global min and max values
    for base_name in base_names:
        for valor_label in valores_label:
            file_name = f"RAS_{valor_label}deg_{base_name}"
            file_path = os.path.join(csv_folder, file_name)
            if os.path.exists(file_path):
                try:
                    # Read the .csv file using pandas
                    data = pd.read_csv(file_path)
                    
                    # Remove rows that do not contain valid numeric values
                    data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    if not data.empty:
                        global_min = min(global_min, data.iloc[:, 0].min())
                        global_max = max(global_max, data.iloc[:, 0].max())
                except Exception as e:
                    print(f"Error processing the file {file_name}: {e}")

    # Then, plot the graphs adjusting the axes
    for base_name in base_names:
        fig = go.Figure()
        for valor_label in valores_label:
            file_name = f"RAS_{valor_label}deg_{base_name}"
            file_path = os.path.join(csv_folder, file_name)
            if os.path.exists(file_path):
                try:
                    # Read the .csv file using pandas
                    data = pd.read_csv(file_path)
                    
                    # Remove rows that do not contain valid numeric values
                    data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    # Check if there are enough data points to plot
                    if data.empty or data.shape[0] < 2:
                        print(f"The file {file_name} does not have enough data to plot.")
                        continue
                    
                    # Plot the CDF
                    fig.add_trace(go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode='lines', name=f'Elevation {valor_label} deg'))
                except Exception as e:
                    print(f"Error processing the file {file_name}: {e}")
        
        # Graph configurations
        fig.update_layout(
            title=f'CDF Plot for {base_name}',
            xaxis_title=xaxis_title,
            yaxis_title='CDF',
            yaxis=dict(tickmode='array', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            xaxis=dict(tickmode='linear', tick0=int(global_min), dtick=passo_xticks),
            legend_title="Labels"
        )
        
        # Show the figure
        fig.show()
        
        # Save the figure
        #base_name_no_ext = os.path.splitext(base_name)[0]  # Remove the .csv extension
        #fig_file_path = os.path.join(figs_folder, f"CDF_Plot_{base_name_no_ext}.png")
        #fig.write_image(fig_file_path)
        #print(f"Figure saved: {fig_file_path}")

def plot_bs_antenna_gain_towards_the_ue(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('BS_antenna_gain_towards_the_UE', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_coupling_loss(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('coupling_loss', label, valores_label, passo_xticks, xaxis_title='Coupling Loss (dB)')

def plot_dl_sinr(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('DL_SINR', label, valores_label, passo_xticks, xaxis_title='DL SINR (dB)')

def plot_dl_snr(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('DL_SNR', label, valores_label, passo_xticks, xaxis_title='DL SNR (dB)')

def plot_dl_throughput(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('DL_throughput', label, valores_label, passo_xticks, xaxis_title='DL Throughput (Mbps)')

def plot_dl_transmit_power(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('DL_transmit_power', label, valores_label, passo_xticks, xaxis_title='DL Transmit Power (dBm)')

def plot_imt_station_antenna_gain_towards_system(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('IMT_station_antenna_gain_towards_system', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_path_loss(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('path_loss', label, valores_label, passo_xticks, xaxis_title='Path Loss (dB)')

def plot_ue_antenna_gain_towards_the_bs(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('UE_antenna_gain_towards_the_BS', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_imt_to_system_path_loss(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('IMT_to_system_path_loss', label, valores_label, passo_xticks, xaxis_title='Path Loss (dB)')

def plot_system_antenna_towards_imt_stations(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('system_antenna_gain_towards_IMT_stations', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_system_inr(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('system_INR', label, valores_label, passo_xticks, xaxis_title='INR (dB)')

def plot_system_interference_power_from_imt_dl(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('system_interference_power_from_IMT_DL', label, valores_label, passo_xticks, xaxis_title='Interference Power (dBm)')

def plot_system_pfd(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('system_PFD', label, valores_label, passo_xticks, xaxis_title='PFD (dBW/m²)')

def plot_inr_samples(label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf('INR_samples', label, valores_label, passo_xticks, xaxis_title='INR Samples (dB)')

# Main function to identify labels and call the appropriate functions
def main():
    valores_label = ['30', '45', '90']
    plot_bs_antenna_gain_towards_the_ue(valores_label=valores_label)
    plot_coupling_loss(valores_label=valores_label)
    plot_dl_sinr(valores_label=valores_label)
    plot_dl_snr(valores_label=valores_label)
    plot_dl_throughput(valores_label=valores_label)
    plot_dl_transmit_power(valores_label=valores_label)
    plot_imt_station_antenna_gain_towards_system(valores_label=valores_label)
    plot_path_loss(valores_label=valores_label)
    plot_ue_antenna_gain_towards_the_bs(valores_label=valores_label)
    plot_imt_to_system_path_loss(valores_label=valores_label)
    plot_system_antenna_towards_imt_stations(valores_label=valores_label)
    plot_system_inr(valores_label=valores_label)
    plot_system_interference_power_from_imt_dl(valores_label=valores_label)
    plot_system_pfd(valores_label=valores_label)
    plot_inr_samples(valores_label=valores_label)

if __name__ == "__main__":
    main()
