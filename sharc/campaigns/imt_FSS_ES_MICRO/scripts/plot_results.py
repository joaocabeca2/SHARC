import os
from pathlib import Path
from sharc.results import Results
# import plotly.graph_objects as go
from sharc.post_processor import PostProcessor

post_processor = PostProcessor()

# Add a legend to results in folder that match the pattern
# This could easily come from a config file
post_processor\
    .add_plot_legend_pattern(
        dir_name_contains="output_DL_LF_50",
        legend="DL - Distância = 1000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_DL_LF_50_DIST_2000",
        legend="DL - Distância = 2000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_DL_LF_50_DIST_5000",
        legend="DL - Distância = 5000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_DL_LF_50_DIST_10000",
        legend="DL - Distância = 10000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_UL_LF_50_DIST_1000",
        legend="UL - Distância = 1000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_UL_LF_50_DIST_2000",
        legend="UL - Distância = 2000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_UL_LF_50_DIST_5000",
        legend="UL - Distância = 5000m"
    ).add_plot_legend_pattern(
        dir_name_contains="output_UL_LF_50_DIST_10000",
        legend="UL - Distância = 10000m"
    )

campaign_base_dir = str((Path(__file__) / ".." / "..").resolve())

many_results = Results.load_many_from_dir(os.path.join(campaign_base_dir, "output"), only_latest=True)
# ^: typing.List[Results]

post_processor.add_results(many_results)

plots = post_processor.generate_cdf_plots_from_results(
    many_results
)

post_processor.add_plots(plots)

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

# Plot every plot:
output_dir = os.path.join(campaign_base_dir, "plots")
os.makedirs(output_dir, exist_ok=True)  # Cria a pasta plots/ se não existir

for idx, plt in enumerate(plots):
    if plt:
        # Atualiza seu layout como você já faz
        plt.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=[1, 0.1, 0.01, 0.001, 0.0001],
                ticktext=["1", "0.1", "0.01", "0.001", "0.0001"],
                type="log",
                range=[-5, 0]
            )
        )
        plt.add_vline(-10.5, line_dash="dash", name="20% criteria x")
        plt.add_vline(-6, line_dash="dash", name="30% criteria x")
        plt.add_hline(0.2, line_dash="dash", name="20% criteria y")
        plt.add_hline(0.0003, line_dash="dash", name="30% criteria y")

        # Salva o plot como imagem
        save_path = os.path.join(output_dir, f"plot_{idx}.png")
        plt.write_image(save_path)

        print(f"Salvou: {save_path}")
        
        # Se ainda quiser mostrar
        #plt.show()

for result in many_results:
    # This generates the mean, median, variance, etc
    stats = PostProcessor.generate_statistics(
        result=result
    ).write_to_results_dir()
    # # do whatever you want here:
    # if "fspl_45deg" in stats.results_output_dir:
    #     get some stat and do something

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
