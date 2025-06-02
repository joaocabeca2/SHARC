import os
from pathlib import Path
from sharc.results import Results
from sharc.post_processor import PostProcessor

auto_open = True

local_dir = os.path.dirname(os.path.abspath(__file__))
campaign_base_dir = str((Path(__file__) / ".." / "..").resolve())

post_processor = PostProcessor()

many_results = Results.load_many_from_dir(os.path.join(campaign_base_dir, "output"),
                                          filter_fn=lambda x: "mss_d2d_to_eess" in x,
                                          only_latest=True)

readable_name = {
    "system_d": "System D",
    "system_b": "System B",
}
def linestyle_getter(results):
    if "system_d" in results.output_directory:
        return "dash"
    return "solid"

post_processor.add_results_linestyle_getter(linestyle_getter)

for sys_name in ["system_d", "system_b"]:
    for elev in [5, 30, 60, 90, "uniform_"]:
        if elev == "uniform_":
            readable_elev = "Elev = Unif. Dist."
        else:
            readable_elev = f"Elev = {elev}º"
        # IMT-MSS-D2D-DL to EESS
        post_processor\
            .add_plot_legend_pattern(
                dir_name_contains=f"{elev}elev_{sys_name}",
                legend=f"{readable_name[sys_name]}, {readable_elev}"
            )
# ^: typing.List[Results]

post_processor.add_results(many_results)

plots = post_processor.generate_cdf_plots_from_results(
    many_results
)

post_processor.add_plots(plots)

# plots = post_processor.generate_ccdf_plots_from_results(
#     many_results
# )

# post_processor.add_plots(plots)

# Add a protection criteria line:
protection_criteria = -154.0  # dB[W/MHz]
perc_time = 0.01
system_dl_interf_power_per_mhz = post_processor.get_plot_by_results_attribute_name("system_dl_interf_power_per_mhz",
                                                                                   plot_type="cdf")
system_dl_interf_power_per_mhz.add_vline(protection_criteria, line_dash="dash", annotation=dict(
    text="Protection Criteria: " + str(protection_criteria) + " dB",
    xref="x", yref="y",
    x=protection_criteria + 0.5, y=0.8,
    font=dict(size=12, color="red")
))
system_dl_interf_power_per_mhz.add_hline(perc_time, line_dash="dash", annotation=dict(
    text="Time Percentage: " + str(perc_time * 100) + "%",
    xref="x", yref="y",
    x=protection_criteria + 0.5, y=perc_time + 0.01,
    font=dict(size=12, color="blue")
))


attributes_to_plot = [
    "imt_system_antenna_gain",
    "imt_system_path_loss",
    "system_dl_interf_power",
    "system_dl_interf_power_per_mhz",
    "system_imt_antenna_gain",
    "system_inr",
]

# for attr in attributes_to_plot:
#     post_processor.get_plot_by_results_attribute_name(attr).show()

for attr in attributes_to_plot:
    file = os.path.join(campaign_base_dir, "output", f"{attr}.html")
    post_processor\
        .get_plot_by_results_attribute_name(attr)\
        .write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)
