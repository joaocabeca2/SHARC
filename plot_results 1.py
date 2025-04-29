import os
from pathlib import Path
from sharc.results import Results
from sharc.post_processor import PostProcessor

import matplotlib.pyplot as plt
import numpy as np

post_processor = PostProcessor()

# Add a legend to results in folder that match the pattern
# This could easily come from a config file
post_processor\
    .add_plot_legend_pattern(
        dir_name_contains="_0km",
        legend="0 km"
    ).add_plot_legend_pattern(
        dir_name_contains="_45km",
        legend="45 km"
    ).add_plot_legend_pattern(
        dir_name_contains="_90km",
        legend="90 km"
    ).add_plot_legend_pattern(
        dir_name_contains="_500km",
        legend="500 km"
    )

campaign_base_dir = str((Path(__file__) / ".." / "..").resolve())

many_results = Results.load_many_from_dir(os.path.join(campaign_base_dir, "output"), only_latest=True)
# ^: typing.List[Results]

post_processor.add_results(many_results)

plots = post_processor.generate_cdf_plots_from_results(
    many_results
)

post_processor.add_plots(plots)

# Setup using matplotlib
for plot in plots:
    # Get x and y data from plotly object
    x_values = []
    y_values = []

    for trace in plot.data:
        x_values.append(np.array(trace['x']))
        y_values.append(np.array(trace['y']))
    
    plt.rcParams["font.family"] = "Arial"
    inr_criteria = np.array([[0.2, -10.5],[5e-4, -1.3]]) # INR thresholds

    if (plot.layout.meta.get('related_results_attribute') == 'system_inr'): # CCDF
        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            plt.semilogy(x, 1-y, label=plot.data[i]['name'])

        cr1 = 'Criteria = ' + str(inr_criteria[0,1]) + ' dB'
        cr2 = 'Criteria = ' + str(inr_criteria[1,1]) + ' dB'

        # Thtresholds
        plt.semilogy(inr_criteria[0,1],inr_criteria[0,0],'.', color='orange')
        plt.semilogy(inr_criteria[1,1],inr_criteria[1,0],'.', color = 'red')        

        plt.axvline(x = inr_criteria[0,1], color="orange", linestyle="--", linewidth=1.0)
        plt.axhline(y = inr_criteria[0,0], color="orange", linestyle="--", linewidth=1.0, label = cr1)
        plt.axvline(x = inr_criteria[1,1], color="red", linestyle="--", linewidth=1.0)
        plt.axhline(y = inr_criteria[1,0], color="red", linestyle="--", linewidth=1.0, label = cr2)   

        #  Cosmetics
        plt.grid(which="major", linestyle="--", linewidth = 0.5, color = 'gray', dashes=[4, 4]) 
        plt.grid(which="minor", linestyle="--", linewidth = 0.2, color = 'gray', dashes=[10, 10]) 
        plt.minorticks_on()

        plt.title("CCDF of System INR")
        plt.xlabel(plot.layout.xaxis.title.text)
        plt.ylabel("P(INR > x)")
        plt.legend()        
        
        # Save as png   
        fname = os.path.join(campaign_base_dir, "output", plot.layout.meta.get('related_results_attribute')) + ".png"
        plt.savefig(fname, format="png", dpi=600)

    else:
        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(zip(x_values, y_values)):            
            plt.plot(x, y, label=plot.data[i]['name'])  # Use the name for labels
        
        plt.grid(which="major", linestyle="--", linewidth = 0.5, color = 'k') 
        plt.grid(which="minor", linestyle="--", linewidth = 0.2, color = 'k') 
        plt.minorticks_on()

        plt.title(plot.layout.title.text)
        plt.xlabel(plot.layout.xaxis.title.text)
        plt.ylabel(plot.layout.yaxis.title.text)
        plt.legend()
        plt.ylim(0, 1)

        # Save as png   
        fname = os.path.join(campaign_base_dir, "output", plot.layout.meta.get('related_results_attribute')) + ".png"
        plt.savefig(fname, format="png", dpi=600)



# # This function aggregates IMT downlink and uplink
# aggregated_results = PostProcessor.aggregate_results(
#     downlink_result=post_processor.get_results_by_output_dir("MHz_60deg_dl"),
#     uplink_result=post_processor.get_results_by_output_dir("MHz_60deg_ul"),
#     ul_tdd_factor=(3, 4),
#     n_bs_sim=7 * 19 * 3 * 3,
#     n_bs_actual=int
# )

# Add a protection criteria line:
# protection_criteria = 160

# post_processor\
#     .get_plot_by_results_attribute_name("system_dl_interf_power")\
#     .add_vline(protection_criteria, line_dash="dash")

# Show a single plot:
# post_processor\
#     .get_plot_by_results_attribute_name("system_dl_interf_power")\
#     .show()

# for result in many_results:
#     # This generates the mean, median, variance, etc
#     stats = PostProcessor.generate_statistics(
#         result=result
#     ).write_to_results_dir()
#     # # do whatever you want here:
#     # if "fspl_45deg" in stats.results_output_dir:
#     #     get some stat and do something

# # example on how to aggregate results and add it to plot:
# dl_res = post_processor.get_results_by_output_dir("1_cluster")
# aggregated_results = PostProcessor.aggregate_results(
#     dl_samples=dl_res.system_dl_interf_power,
#     ul_samples=ul_res.system_ul_interf_power,
#     ul_tdd_factor=0.75,
#     n_bs_sim=1 * 19 * 3 * 3,
#     n_bs_actual=7 * 19 * 3 * 3
# )

# relevant = post_processor\
#     .get_plot_by_results_attribute_name("system_ul_interf_power")

# aggr_x, aggr_y = PostProcessor.cdf_from(aggregated_results)

# relevant.add_trace(
#     go.Scatter(x=aggr_x, y=aggr_y, mode='lines', name='Aggregate interference',),
# )

# relevant.show()
