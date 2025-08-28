from sharc.results import Results

from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import os
import numpy as np
import scipy
import typing
import pathlib
import pathlib


class FieldStatistics:
    """
    Stores statistical properties of a data field, such as mean, median, and variance.
    """
    field_name: str
    median: float
    mean: float
    variance: float
    confidence_interval: (float, float)
    standard_deviation: float

    def load_from_sample(
        self, field_name: str, sample: list[float], *, confidence=0.95
    ) -> "FieldStatistics":
        """
        Compute statistics from a sample and store them in the object.
        """
        self.field_name = field_name
        self.median = np.median(sample)
        self.mean = np.mean(sample)
        self.variance = np.var(sample)
        self.standard_deviation = np.std(sample)
        # @important TODO: check if using t distribution here is correct
        self.confidence_interval = scipy.stats.norm.interval(
            confidence, loc=self.mean, scale=self.standard_deviation
        )
        return self

    def __str__(self):
        attr_names = filter(
            lambda x: x != "field_name"
            and not x.startswith("__")
            and not callable(getattr(self, x)),
            dir(self),
        )
        readable_attrs = "\n".join(
            list(
                map(
                    lambda attr_name: f"\t{attr_name}: {getattr(self, attr_name)}",
                    attr_names,
                )
            )
        )
        return f"""{self.field_name}:
{readable_attrs}"""


class ResultsStatistics:
    """Class that stores and computes statistics for results fields."""

    fields_statistics: list[FieldStatistics]
    results_output_dir: str = "default_output"

    def load_from_results(self, result: Results) -> "ResultsStatistics":
        """Load all relevant attributes from result and generate their statistics."""
        self.results_output_dir = result.output_directory
        self.fields_statistics = []
        attr_names = result.get_relevant_attributes()
        for attr_name in attr_names:
            samples = getattr(result, attr_name)
            if len(samples) == 0:
                continue
            self.fields_statistics.append(
                FieldStatistics().load_from_sample(attr_name, samples)
            )

        return self

    def write_to_results_dir(
            self,
            filename="stats.txt") -> "ResultsStatistics":
        """Write statistics file to the same directory of the results loaded into this class."""
        with open(os.path.join(self.results_output_dir, filename), "w") as f:
            f.write(str(self))

        return self

    def get_stat_by_name(
            self, field_name: str) -> typing.Union[None, FieldStatistics]:
        """Get a single field's statistics by its name."""
        stats_found = filter(
            lambda field_stat: field_stat.field_name == field_name,
            self.fields_statistics)

        if len(stats_found) > 1:
            raise Exception(
                f"ResultsStatistics.get_stat_by_name found more than one statistic by the field name '{field_name}'\n" +
                "You probably loaded more than one result to the same ResultsStatistics object")

        if len(stats_found) == 0:
            return None

        return stats_found[0]

    def __str__(self):
        """Return a string representation of the ResultsStatistics object."""
        return f"[{self.results_output_dir}]\n{'\n'.join(list(map(str, self.fields_statistics)))}"


@dataclass
class PostProcessor:
    """
    PostProcessor provides utilities for plotting, aggregating, and analyzing simulation results,
    including CDF/CCDF plot generation, statistics calculation, and plot saving.
    """
    IGNORE_FIELD = {
        "title": "ANTES NAO PLOTAVAMOS ISSO, ENTÃO CONTINUA SEM PLOTAR",
        "x_label": "",
    }
    # TODO: move units in the result to the Results class instead of Plot Info?
    # TODO: rename x_label to something else when making plots other than cdf
    RESULT_FIELDNAME_TO_PLOT_INFO = {
        "imt_ul_tx_power_density": {
            "x_label": "Transmit power density [dBm/Hz]",
            "title": "[IMT] UE transmit power density",
        },
        "imt_ul_tx_power": {
            "x_label": "Transmit power [dBm]",
            "title": "[IMT] UE transmit power",
        },
        "imt_ul_sinr_ext": {
            "x_label": "SINR [dB]",
            "title": "[IMT] UL SINR with external interference",
        },
        "imt_ul_snr": {
            "title": "[IMT] UL SNR",
            "x_label": "SNR [dB]",
        },
        "imt_ul_inr": {
            "title": "[IMT] UL interference-to-noise ratio",
            "x_label": "$I/N$ [dB]",
        },
        "imt_ul_sinr": {
            "x_label": "SINR [dB]",
            "title": "[IMT] UL SINR",
        },
        "imt_system_build_entry_loss": {
            "x_label": "Building entry loss [dB]",
            "title": "[SYS] IMT to system building entry loss",
        },
        "imt_ul_tput_ext": {
            "title": "[IMT] UL throughput with external interference",
            "x_label": "Throughput [bits/s/Hz]",
        },
        "imt_ul_tput": {
            "title": "[IMT] UL throughput",
            "x_label": "Throughput [bits/s/Hz]",
        },
        "imt_path_loss": {
            "title": "[IMT] path loss",
            "x_label": "Path loss [dB]",
        },
        "imt_coupling_loss": {
            "title": "[IMT] coupling loss",
            "x_label": "Coupling loss [dB]",
        },
        "imt_bs_antenna_gain": {
            "x_label": "Antenna gain [dBi]",
            "title": "[IMT] BS antenna gain towards the UE",
        },
        "imt_ue_antenna_gain": {
            "x_label": "Antenna gain [dBi]",
            "title": "[IMT] UE antenna gain towards the BS",
        },
        "system_imt_antenna_gain": {
            "x_label": "Antenna gain [dBi]",
            "title": "[SYS] system antenna gain towards IMT stations",
        },
        "imt_system_antenna_gain": {
            "x_label": "Antenna gain [dBi]",
            "title": "[IMT] IMT station antenna gain towards system",
        },
        "imt_system_path_loss": {
            "x_label": "Path Loss [dB]",
            "title": "[SYS] IMT to system path loss",
        },
        "sys_to_imt_coupling_loss": {
            "x_label": "Coupling Loss [dB]",
            "title": "[SYS] IMT to system coupling loss",
        },
        "system_dl_interf_power": {
            "x_label": "Interference Power [dB]",
            "title": "[SYS] system interference power from IMT DL",
        },
        "imt_system_diffraction_loss": {
            "x_label": "Building entry loss [dB]",
            "title": "[SYS] IMT to system diffraction loss",
        },
        "imt_dl_sinr_ext": {
            "x_label": "SINR [dB]",
            "title": "[IMT] DL SINR with external interference",
        },
        "imt_dl_sinr": {
            "x_label": "SINR [dB]",
            "title": "[IMT] DL SINR",
        },
        "imt_dl_snr": {
            "title": "[IMT] DL SNR",
            "x_label": "SNR [dB]",
        },
        "imt_dl_inr": {
            "title": "[IMT] DL interference-to-noise ratio",
            "x_label": "$I/N$ [dB]",
        },
        "imt_dl_tput_ext": {
            "title": "[IMT] DL throughput with external interference",
            "x_label": "Throughput [bits/s/Hz]",
        },
        "imt_dl_tput": {
            "title": "[IMT] DL throughput",
            "x_label": "Throughput [bits/s/Hz]",
        },
        "system_ul_interf_power": {
            "title": "[SYS] system interference power from IMT UL",
            "x_label": "Interference Power [dBm/BMHz]",
        },
        "system_ul_interf_power_per_mhz": {
            "title": "[SYS] system interference PSD from IMT UL",
            "x_label": "Interference Power [dBm/MHz]",
        },
        "system_dl_interf_power_per_mhz": {
            "title": "[SYS] system interference PSD from IMT DL",
            "x_label": "Interference Power [dB/MHz]",
        },
        "system_inr": {
            "title": "[SYS] system INR",
            "x_label": "INR [dBm]",
        },
        "system_pfd": {
            "title": "[SYS] system PFD",
            "x_label": "PFD [dBm/m^2]",
        },
        "imt_dl_tx_power": {
            "x_label": "Transmit power [dBm]",
            "title": "[IMT] DL transmit power",
        },
        "imt_dl_pfd_external": {
            "title": "[IMT] DL external Power Flux Density (PFD) ",
            "x_label": "PFD [dBW/m²/MHz]",
        },
        "imt_dl_pfd_external_aggregated": {
            "title": "[IMT] Aggregated DL external Power Flux Density (PFD)",
            "x_label": "PFD [dBW/m²/MHz]",
        },
        # these ones were not plotted already, so will continue to not be
        # plotted:
        "imt_dl_tx_power_density": IGNORE_FIELD,
        "system_ul_coupling_loss": IGNORE_FIELD,
        "system_dl_coupling_loss": IGNORE_FIELD,
        "system_rx_interf": IGNORE_FIELD,
    }

    plot_legend_patterns: list = field(default_factory=list)
    legends_generator = None
    linestyle_getter = None

    plots: list[go.Figure] = field(default_factory=list)
    results: list[Results] = field(default_factory=list)

    def add_plot_legend_generator(
        self, generator
    ) -> "PostProcessor":
        """
        You can either add a plot generator or many plot legend patterns.
        A generator is much more flexible.
        """
        if self.legends_generator is not None:
            raise ValueError("Can only have one legends generator at a time")
        self.legends_generator = generator

    def add_results_linestyle_getter(
        self, getter
    ) -> None:
        """
        When plotting, this function will be called for each result to decide
        on the linestyle used
        """
        if self.linestyle_getter is not None:
            raise ValueError(
                "You are trying to set PostProcessor.linestyle_getter twice!")
        self.linestyle_getter = getter

    def add_plot_legend_pattern(
        self, *, dir_name_contains: str, legend: str
    ) -> "PostProcessor":
        """Add a plot legend pattern for directory name matching and return self."""
        self.plot_legend_patterns.append(
            {"dir_name_contains": dir_name_contains, "legend": legend}
        )
        self.plot_legend_patterns.sort(
            key=lambda p: -len(p["dir_name_contains"]))

        return self

    def get_results_possible_legends(self, result: Results) -> list[dict]:
        """
        You get a list with dicts tha have at least { "legend": str } in them.
        They may also have { "dir_name_contains": str }
        """
        possible = list(
            filter(
                lambda pl: pl["dir_name_contains"]
                in os.path.basename(result.output_directory),
                self.plot_legend_patterns,
            )
        )

        if len(possible) == 0 and self.legends_generator is not None:
            return [{"legend": self.legends_generator(
                os.path.basename(result.output_directory))}]

        return possible

    def generate_cdf_plots_from_results(
        self, results: list[Results], *, n_bins=None
    ) -> list[go.Figure]:
        """
        Generates CDF (Cumulative Distribution Function) plots from a list of Results objects.

        This method processes each Results object, extracts relevant attributes, and generates
        CDF plots for each attribute using Plotly. Each plot is configured with appropriate
        titles, axis labels, and legends. The method supports custom line styles and colors
        for different result sets.

        Args:
            results (list[Results]): A list of Results objects containing data to plot.
            n_bins (Optional[int], optional): Number of bins to use for the CDF calculation.
                If None, a default binning strategy is used.

        Returns:
            list[go.Figure]: A list of Plotly Figure objects, each representing a CDF plot
                for a relevant attribute found in the Results objects.
        """
        figs: dict[str, list[go.Figure]] = {}
        COLORS = DEFAULT_PLOTLY_COLORS

        linestyle_color = {}

        # Sort based on path name - TODO: sort alphabeticaly by legend
        results.sort(key=lambda r: r.output_directory)
        for res in results:
            if self.linestyle_getter is not None:
                linestyle = self.linestyle_getter(res)
            else:
                linestyle = "solid"

            if linestyle not in linestyle_color:
                linestyle_color[linestyle] = 0

            possible_legends_mapping = self.get_results_possible_legends(res)

            if len(possible_legends_mapping):
                legend = possible_legends_mapping[0]["legend"]
            else:
                legend = res.output_directory

            attr_names = res.get_relevant_attributes()

            for attr_name in attr_names:
                attr_val = getattr(res, attr_name)
                if not len(attr_val):
                    continue
                if attr_name not in PostProcessor.RESULT_FIELDNAME_TO_PLOT_INFO:
                    print(
                        f"[WARNING]: {attr_name} is not a plottable field, because it does not have a configuration set on PostProcessor.")
                    continue
                attr_plot_info = PostProcessor.RESULT_FIELDNAME_TO_PLOT_INFO[attr_name]
                if attr_plot_info == PostProcessor.IGNORE_FIELD:
                    print(
                        f"[WARNING]: {attr_name} is currently being ignored on plots.")
                    continue
                if attr_name not in figs:
                    figs[attr_name] = go.Figure()
                    figs[attr_name].update_layout(
                        title=f'CDF Plot for {
                            attr_plot_info["title"]}',
                        xaxis_title=attr_plot_info["x_label"],
                        yaxis_title="P(X <= x)",
                        yaxis=dict(
                            tickmode="array",
                            tickvals=[
                                0,
                                0.25,
                                0.5,
                                0.75,
                                1]),
                        xaxis=dict(
                            tickmode="linear",
                            dtick=5),
                        legend_title="Labels",
                        meta={
                            "related_results_attribute": attr_name,
                            "plot_type": "cdf"},
                    )

                # TODO: take this fn as argument, to plot more than only cdf's
                x, y = PostProcessor.cdf_from(attr_val, n_bins=n_bins)

                fig = figs[attr_name]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{legend}",
                        line=dict(color=COLORS[linestyle_color[linestyle]], dash=linestyle)
                    ),
                )

            linestyle_color[linestyle] += 1
            if linestyle_color[linestyle] >= len(COLORS):
                linestyle_color[linestyle] = 0

        return figs.values()

    def generate_ccdf_plots_from_results(
        self, results: list[Results], *, n_bins=None, cutoff_percentage=0.001, logy=True
    ) -> list[go.Figure]:
        """
        Generates CCDF (Complementary Cumulative Distribution Function) plots from a list of Results objects.

        This method processes each Results object, extracts relevant attributes, and generates
        CCDF plots for each attribute using Plotly. Each plot is configured with appropriate
        titles, axis labels, and legends. The method supports log-scale y-axes and custom cutoff percentages.

        Args:
            results (list[Results]): A list of Results objects containing data to plot.
            n_bins (Optional[int], optional): Number of bins to use for the CCDF calculation.
                If None, a default binning strategy is used.
            cutoff_percentage (float, optional): The minimum probability to display on the y-axis.
            logy (bool, optional): Whether to use a logarithmic scale for the y-axis.

        Returns:
            list[go.Figure]: A list of Plotly Figure objects, each representing a CCDF plot
                for a relevant attribute found in the Results objects.
        """
        figs: dict[str, list[go.Figure]] = {}
        COLORS = DEFAULT_PLOTLY_COLORS

        linestyle_color = {}

        results.sort(key=lambda r: r.output_directory)
        for res in results:
            if self.linestyle_getter is not None:
                linestyle = self.linestyle_getter(res)
            else:
                linestyle = "solid"

            if linestyle not in linestyle_color:
                linestyle_color[linestyle] = 0

            possible_legends_mapping = self.get_results_possible_legends(res)

            if len(possible_legends_mapping):
                legend = possible_legends_mapping[0]["legend"]
            else:
                legend = res.output_directory

            attr_names = res.get_relevant_attributes()

            next_tick = 1
            ticks_at = []
            while next_tick > cutoff_percentage:
                ticks_at.append(next_tick)
                next_tick /= 10
            ticks_at.append(cutoff_percentage)
            ticks_at.reverse()

            for attr_name in attr_names:
                attr_val = getattr(res, attr_name)
                if not len(attr_val):
                    continue
                if attr_name not in PostProcessor.RESULT_FIELDNAME_TO_PLOT_INFO:
                    print(
                        f"[WARNING]: {attr_name} is not a plottable field, because it does not have a configuration set on PostProcessor.")
                    continue
                attr_plot_info = PostProcessor.RESULT_FIELDNAME_TO_PLOT_INFO[attr_name]
                if attr_plot_info == PostProcessor.IGNORE_FIELD:
                    print(
                        f"[WARNING]: {attr_name} is currently being ignored on plots.")
                    continue
                if attr_name not in figs:
                    figs[attr_name] = go.Figure()
                    figs[attr_name].update_layout(
                        title=f'CCDF Plot for {
                            attr_plot_info["title"]}',
                        xaxis_title=attr_plot_info["x_label"],
                        yaxis_title="P(X > x)",
                        yaxis=dict(
                            tickmode="array",
                            tickvals=ticks_at,
                            type="log",
                            range=[
                                np.log10(cutoff_percentage),
                                0]),
                        xaxis=dict(
                            tickmode="linear",
                            dtick=5),
                        legend_title="Labels",
                        meta={
                            "related_results_attribute": attr_name,
                            "plot_type": "ccdf"},
                    )

                # TODO: take this fn as argument, to plot more than only cdf's
                x, y = PostProcessor.ccdf_from(attr_val, n_bins=n_bins)

                fig = figs[attr_name]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{legend}",
                        line=dict(color=COLORS[linestyle_color[linestyle]], dash=linestyle)
                    ),
                )
                # A trick to plog semi-logy plots with better scientific aspect.
                # I should have left it to the user to decide if they want logy or not,
                # but I think it is better to have it by default.
                if logy:
                    fig.update_yaxes(type="log")
                    yticks = []
                    n_right_zeros = -int(np.floor(np.log10(cutoff_percentage)))
                    for i in range(n_right_zeros, 0, -1):
                        for j in range(1, 10):
                            yticks.append(j * (10**(-i)))
                    yticks = yticks + [1.0]
                    major_yticks = [10**(-i) for i in range(n_right_zeros)]
                    ytick_text = [
                        str(v) if v in major_yticks else "" for v in yticks]
                    fig.update_yaxes(tickvals=yticks, ticktext=ytick_text)

            linestyle_color[linestyle] += 1
            if linestyle_color[linestyle] >= len(COLORS):
                linestyle_color[linestyle] = 0

        return figs.values()

    def add_plots(self, plots: list[go.Figure]) -> None:
        """Add a list of plotly Figure objects to the PostProcessor."""
        self.plots.extend(plots)

    def add_results(self, results: list[Results]) -> None:
        """Add a list of Results objects to the PostProcessor."""
        self.results.extend(results)

    def get_results_by_output_dir(
            self,
            dir_name_contains: str,
            *,
            single_result=True):
        """
        Get results whose output directory contains the given string.
        If single_result is True, return the first match; otherwise, return all matches.
        """
        filtered_results = list(
            filter(
                lambda res: dir_name_contains in os.path.basename(
                    res.output_directory),
                self.results,
            ))
        if len(filtered_results) == 0:
            raise ValueError(
                f"Could not find result that contains '{dir_name_contains}'"
            )

        if len(filtered_results) > 1:
            raise ValueError(
                f"There is more than one possible result with pattern '{dir_name_contains}'")

        return filtered_results[0]

    def get_plot_by_results_attribute_name(
            self, attr_name: str, *, plot_type="cdf") -> go.Figure:
        """
        You can get a plot using an attribute name from Results.
        See Results class to check what attributes exist.
        plot_type: 'cdf', 'ccdf'
        """
        filtered = list(
            filter(
                lambda x: x.layout.meta["related_results_attribute"] == attr_name and x.layout.meta["plot_type"] == plot_type,
                self.plots,
            ))

        if 0 == len(filtered):
            return None

        return filtered[0]

    @staticmethod
    def aggregate_results(
        *,
        dl_samples: list[float],
        ul_samples: list[float],
        ul_tdd_factor: float,
        n_bs_sim: int,
        n_bs_actual: int,
        random_number_gen=np.random.RandomState(31),
    ):
        """
        The method was adapted from document 'TG51_201805_E07_FSS_Uplink_ study 48GHz_GSMA_v1.5.pdf',
        a document created for Task Group 5/1.
        This is used to aggregate both uplink and downlink interference towards another system
            into a result that makes more sense to the case study.
        Inputs:
            downlink/uplink_result: list[float]
                Samples that should be aggregated.
            ul_tdd_factor: float
                The tdd ratio that uplink is activated for.
            n_bs_sim: int
                Number of simulated base stations.
                Should probably be 7 * 19 * 3 * 3 or 1 * 19 * 3 * 3
            n_bs_actual: int
                The number of base stations the study wants to have conclusions for.
            random_number_gen: np.random.RandomState
                Since this methods uses another montecarlo to aggregate results,
                it needs a random number generator
        """
        if ul_tdd_factor > 1 or ul_tdd_factor < 0:
            raise ValueError(
                "PostProcessor.aggregate_results() was called with invalid ul_tdd_factor parameter." +
                f"ul_tdd_factor must be in interval [0, 1], but is {ul_tdd_factor}")

        segment_factor = round(n_bs_actual / n_bs_sim)

        dl_tdd_factor = 1 - ul_tdd_factor

        if ul_tdd_factor == 0:
            n_aggregate = len(dl_samples)
        elif dl_tdd_factor == 0:
            n_aggregate = len(ul_samples)
        else:
            n_aggregate = min(len(ul_samples), len(dl_samples))

        aggregate_samples = np.empty(n_aggregate)

        for i in range(n_aggregate):
            # choose S random samples
            ul_random_indexes = np.floor(
                random_number_gen.random(size=segment_factor)
                * len(ul_samples)
            )
            dl_random_indexes = np.floor(
                random_number_gen.random(size=segment_factor)
                * len(dl_samples)
            )
            aggregate_samples[i] = 0

            if ul_tdd_factor:
                for j in ul_random_indexes:  # random samples
                    aggregate_samples[i] += (
                        np.power(10, ul_samples[int(j)] / 10) * ul_tdd_factor
                    )

            if dl_tdd_factor:
                for j in dl_random_indexes:  # random samples
                    aggregate_samples[i] += (
                        np.power(10, dl_samples[int(j)] / 10) * dl_tdd_factor
                    )

            # convert back to dB or dBm (as was previously)
            aggregate_samples[i] = 10 * np.log10(
                aggregate_samples[i]
            )

        return aggregate_samples

    @staticmethod
    def cdf_from(data: list[float], *, n_bins=None) -> (list[float], list[float]):
        """
        Takes a dataset and returns both axis of a cdf (x, y)
        """
        values, base = np.histogram(
            data,
            bins=n_bins if n_bins is not None else "auto",
        )
        cumulative = np.cumsum(values)
        x = base[:-1]
        #y = cumulative / cumulative[-1] #cdf
        y = 1 - (cumulative / cumulative[-1]) #ccdf
        return (x, y)

    @staticmethod
    def ccdf_from(data: list[float], *, n_bins=None) -> (list[float], list[float]):
        """
        Takes a dataset and returns both axis of a ccdf (x, y)
        """
        x, y = PostProcessor.cdf_from(data, n_bins=n_bins)

        return (x, 1 - y)

    @staticmethod
    def save_plots(
        dir: str,
        plots: list[go.Figure],
        *,
        width=1200,
        height=800
    ) -> None:
        """
        dir: A directory path on which to save the plot files
        plots: Figures to save. They are saved by their name
        """
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        for plot in plots:
            # TODO: check if reset to previous state is functional
            # so much state used in this post processor, should def. migrate
            # post processing to Haskell (obv. not rly)
            prev_autosize = plot.layout.autosize
            prev_width = plot.layout.width
            prev_height = plot.layout.height

            plot.update_layout(
                autosize=False,
                width=width,
                height=height
            )

            plot.write_image(
                os.path.join(
                    dir, f"{
                        plot.layout.title.text}.jpg"))

            plot.update_layout(
                autosize=prev_autosize,
                width=prev_width,
                height=prev_height
            )

    @staticmethod
    def generate_statistics(result: Results) -> ResultsStatistics:
        """Generate statistics for a Results object."""
        return ResultsStatistics().load_from_results(result)

    @staticmethod
    def generate_sample_statistics(
            fieldname: str,
            sample: list[float]) -> ResultsStatistics:
        """Generate statistics for a sample field."""
        return FieldStatistics().load_from_sample(fieldname, sample)
