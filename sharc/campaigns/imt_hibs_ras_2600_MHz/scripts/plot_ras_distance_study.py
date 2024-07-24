import os
import sys
from convert_txt_to_csv import save_to_csv
from plot_cdf import *

# Assuming convert_txt_to_csv.py and plot_cdf.py are in the same directory
sys.path.append(os.path.dirname(__file__))

def main():
    distances = ["0", "45", "90", "500", "5000"]
    labels = [dist + "km_2024-07-24_01" for dist in distances]
    labels2 = [dist + "km" for dist in distances]
    
    # Run the save_to_csv function for each label
    for label, label2 in zip(labels, labels2):
        print(f"Running save_to_csv for label: {label}")
        save_to_csv(label, label2)
        
    # Run the plot functions with the list of label2
    print("Running plot functions for all labels")
    valores_label = distances
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
