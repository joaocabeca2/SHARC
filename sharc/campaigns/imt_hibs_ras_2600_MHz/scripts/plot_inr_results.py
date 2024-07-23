# Script for ploting resutls for the imt_hins_ras_2600MHz campaign
import os
import pandas as pd
import matplotlib.pyplot as plt

my_path = os.path.dirname(os.path.abspath(__file__))

campaing_labels = [('01', )]

plt.figure(figsize=(12, 8))
plt.title("CDF of INR - IMT-NTN DOWNLINK to RAS - 2680MHz - 20km altitude")
plt.ylabel('Probability of P < X', fontsize=10, color='black')
plt.xlabel('INR[dBW/10MHz]', fontsize=10, color='black')
plt.ylim((0.0, 1.0))
plt.axvline(x = -208.0, color = 'b', label = 'Long-term Protection Criteria')
axs = plt.subplot(1, 1, 1)
axs.grid()
plot_labels = ["Long-term Protection Criteria", "0km", "45km", "90km", "500km"]

output_folder = os.path.join(my_path, "../output/")
os.walk(output_folder)
for _, dirnames, _ in os.walk(output_folder):
    # output_folder = f"../output/output_imt_hibs_ras_2600_MHz_0km_2024-07-22_0{i}"
    # Load the results for 0km case
    # interf_power_from_imt_file = \
    #     os.path.join(my_path, (f"{output_folder}/SYS_CDF_of_system_interference_power_from_IMT_DL.txt"))
    for output_dir in dirnames:
        interf_power_from_imt_file = os.path.join(output_folder, output_dir, "SYS_CDF_of_system_interference_power_from_IMT_DL.txt")
        interf_power_cdf = pd.read_csv(interf_power_from_imt_file, skiprows=1, sep='	',header=None)
        # axs.ecdf(dl_inr_samples[1])
        axs.plot(interf_power_cdf[0], interf_power_cdf[1])

plt.legend(plot_labels)
# fig_file_name = os.path.join(my_path, ("../output/CDF_of_RAS_interference_power_from_IMT_DL_2600MHz_at_0km_altitude"))
# plt.savefig(fig_file_name)
plt.show()
