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
post_processor\
    .add_plot_legend_pattern(
        dir_name_contains="mss_d2d_to_eess",
        legend="IMT-MSS-D2D-DL to EESS(s-E)"
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
# protection_criteria = -6
system_inr = post_processor.get_plot_by_results_attribute_name("system_inr", plot_type="cdf")
# imt_dl_inr.add_vline(protection_criteria, line_dash="dash", annotation=dict(
#     text="Protection Criteria: " + str(protection_criteria) + " dB",
#     xref="x", yref="y",
#     x=protection_criteria + 0.5, y=0.8,
#     font=dict(size=12, color="red")
# ))
# imt_dl_inr.update_layout(template="plotly_white")
# imt_ul_inr = post_processor.get_plot_by_results_attribute_name("imt_ul_inr", plot_type="cdf")
# imt_ul_inr.add_vline(protection_criteria, line_dash="dash")

# Combine INR plots into one:

# for trace in imt_ul_inr.data:
#     imt_dl_inr.add_trace(trace)

# Update layout if needed
system_inr.update_layout(title_text="CDF Plot EESS(s-E) INR",
                         xaxis_title="IMT INR [dB]",
                         yaxis_title="Cumulative Probability",
                         legend_title="Legend")

file = os.path.join(campaign_base_dir, "output", "system_inr.html")
system_inr.write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)

# file = os.path.join(campaign_base_dir, "output", "imt_system_antenna_gain.html")
# imt_system_antenna_gain = post_processor.get_plot_by_results_attribute_name("imt_system_antenna_gain")
# imt_system_antenna_gain.write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)

# file = os.path.join(campaign_base_dir, "output", "system_imt_antenna_gain.html")
# system_imt_antenna_gain = post_processor.get_plot_by_results_attribute_name("system_imt_antenna_gain")
# system_imt_antenna_gain.write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)

# file = os.path.join(campaign_base_dir, "output", "sys_to_imt_coupling_loss.html")
# imt_system_path_loss = post_processor.get_plot_by_results_attribute_name("imt_system_path_loss")
# imt_system_path_loss.write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)

# for result in many_results:
#     # This generates the mean, median, variance, etc
#     stats = PostProcessor.generate_statistics(
#         result=result
#     ).write_to_results_dir()
