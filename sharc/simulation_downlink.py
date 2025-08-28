# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:06:41 2017

@author: edgar
"""

import numpy as np
import math
import warnings

from sharc.simulation import Simulation
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.parameters.constants import BOLTZMANN_CONSTANT

warn = warnings.warn


class SimulationDownlink(Simulation):
    """
    Implements the flowchart of simulation downlink method
    """

    def __init__(self, parameters: Parameters, parameter_file: str):
        """Initialize the SimulationDownlink with parameters and parameter file.

        Parameters
        ----------
        parameters : Parameters
            Simulation parameters object.
        parameter_file : str
            Path to the parameter file.
        """
        super().__init__(parameters, parameter_file)

    def snapshot(self, *args, **kwargs):
        """Run a simulation snapshot for the downlink scenario.

        Parameters
        ----------
        *args : tuple
            Positional arguments (unused).
        **kwargs : dict
            Keyword arguments, must include 'write_to_file', 'snapshot_number', and 'seed'.
        """
        write_to_file = kwargs["write_to_file"]
        snapshot_number = kwargs["snapshot_number"]
        seed = kwargs["seed"]

        random_number_gen = np.random.RandomState(seed)

        # In case of hotspots, base stations coordinates have to be calculated
        # on every snapshot. Anyway, let topology decide whether to calculate
        # or not
        num_stations_before = self.topology.num_base_stations

        self.topology.calculate_coordinates(random_number_gen)

        if num_stations_before != self.topology.num_base_stations:
            self.initialize_topology_dependant_variables()

        # Create the base stations (remember that it takes into account the
        # network load factor)
        self.bs = StationFactory.generate_imt_base_stations(
            self.parameters.imt,
            # TODO: remove this:
            self.parameters.imt.bs.antenna.array,
            self.topology, random_number_gen,
        )

        # Create the other system (FSS, HAPS, etc...)
        self.system = StationFactory.generate_system(
            self.parameters, self.topology, random_number_gen,
            geometry_converter=self.geometry_converter
        )

        # Create IMT user equipments
        self.ue = StationFactory.generate_imt_ue(
            self.parameters.imt,
            # TODO: remove this:
            self.parameters.imt.ue.antenna.array,
            self.topology, random_number_gen,
        )

        # self.plot_scenario()

        self.connect_ue_to_bs()
        self.select_ue(random_number_gen)
        self.scheduler()
        self.power_control()

        if self.parameters.imt.interfered_with:
            # Execute this piece of code if the other system generates
            # interference into IMT

            # Calculate coupling loss after beams are created
            self.coupling_loss_imt = self.calculate_intra_imt_coupling_loss(
                self.ue, self.bs,
            )
            self.calculate_sinr()
            self.calculate_sinr_ext()
        else:
            # Execute this piece of code if IMT generates interference into
            # the other system

            # If the intra SINR calculation is disabled, we do not calculate
            # the SINR for the IMT UEs, but we still calculate the external
            # interference.
            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.coupling_loss_imt = self.calculate_intra_imt_coupling_loss(
                    self.ue, self.bs,
                )
                self.calculate_sinr()
            self.calculate_external_interference()

        self.collect_results(write_to_file, snapshot_number)

    def finalize(self, *args, **kwargs):
        """
        Finalize the simulation and notify observers with the results.

        Parameters
        ----------
        *args : tuple
            Positional arguments (unused).
        **kwargs : dict
            Keyword arguments (unused).
        """
        self.notify_observers(source=__name__, results=self.results)

    def power_control(self):
        """
        Apply downlink power control algorithm to distribute power among selected UEs.
        """
        # Currently, the maximum transmit power of the base station is equaly
        # divided among the selected UEs
        total_power = self.parameters.imt.bs.conducted_power \
            + self.bs_power_gain
        tx_power = total_power - 10 * math.log10(self.parameters.imt.ue.k)
        # calculate transmit powers to have a structure such as
        # {bs_1: [pwr_1, pwr_2,...], ...}, where bs_1 is the base station id,
        # pwr_1 is the transmit power from bs_1 to ue_1, pwr_2 is the transmit
        # power from bs_1 to ue_2, etc
        bs_active = np.where(self.bs.active)[0]
        self.bs.tx_power = dict(
            [(bs, tx_power * np.ones(self.parameters.imt.ue.k)) for bs in bs_active])

        # Update the spectral mask
        if self.adjacent_channel:
            self.bs.spectral_mask.set_mask(p_tx=total_power)

    def calculate_sinr(self):
        """
        Calculates the downlink SINR for each UE.
        """
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            ue = self.link[bs]
            self.ue.rx_power[ue] = self.bs.tx_power[bs] - \
                self.coupling_loss_imt[bs, ue]

            # create a list with base stations that generate interference in
            # ue_list
            bs_interf = [b for b in bs_active if b not in [bs]]

            # calculate intra system interference
            for bi in bs_interf:
                interference = self.bs.tx_power[bi] - \
                    self.coupling_loss_imt[bi, ue]

                self.ue.rx_interference[ue] = 10 * np.log10(np.power(
                    10, 0.1 * self.ue.rx_interference[ue]) + np.power(10, 0.1 * interference), )

        # Thermal noise in dBm
        self.ue.thermal_noise = \
            10 * math.log10(BOLTZMANN_CONSTANT * self.parameters.imt.noise_temperature * 1e3) + \
            10 * np.log10(self.ue.bandwidth * 1e6) + \
            self.ue.noise_figure

        self.ue.total_interference = \
            10 * np.log10(
                np.power(10, 0.1 * self.ue.rx_interference) +
                np.power(10, 0.1 * self.ue.thermal_noise),
            )

        self.ue.sinr = self.ue.rx_power - self.ue.total_interference
        self.ue.snr = self.ue.rx_power - self.ue.thermal_noise

    def calculate_sinr_ext(self):
        """
        Calculates the downlink SINR and INR for each UE taking into account the
        interference that is generated by the other system into IMT system.
        """
        if self.co_channel or (
            self.adjacent_channel and self.param_system.adjacent_ch_emissions != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system,
                self.ue,
                is_co_channel=True,
            )
        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.ue,
                    is_co_channel=False,
                )

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        active_sys = np.where(self.system.active)[0]

        # All UEs are active on an active BS
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            ue = self.link[bs]

            # Get the weight factor for the system overlaping bandwidth in each
            # UE band.
            weights = self.calculate_bw_weights(
                self.ue.bandwidth[ue],
                self.ue.center_freq[ue],
                float(self.param_system.bandwidth),
                float(self.param_system.frequency),
            )

            in_band_interf_power = -500.
            if self.co_channel:
                # Inteferer transmit power in dBm over the overlapping band
                # (MHz) with UEs.
                if self.overlapping_bandwidth > 0:
                    # in_band_interf_power = self.param_system.tx_power_density + \
                    #     10 * np.log10(self.overlapping_bandwidth * 1e6) + 30
                    in_band_interf_power = \
                        self.param_system.tx_power_density + 10 * np.log10(
                            self.ue.bandwidth[ue, np.newaxis] * 1e6
                        ) + 10 * np.log10(weights)[:, np.newaxis] - \
                        self.coupling_loss_imt_system[ue, :][:, active_sys]

            oob_power = np.resize(-500., (len(ue), 1))
            if self.adjacent_channel:
                # emissions outside of tx bandwidth and inside of rx bw
                # due to oob emissions on tx side
                tx_oob = np.resize(-500., len(ue))

                # emissions outside of rx bw and inside of tx bw
                # due to non ideal filtering on rx side
                # will be the same for all UE's, only considering
                rx_oob = np.resize(-500., len(ue))

                # TODO: M.2101 states that:
                # "The ACIR value should be calculated based on per UE allocated number of resource blocks"

                # should we actually implement that for ACS since the receiving
                # filter is fixed?

                # or maybe ignore ACS altogether (ACS = inf)? If we consider only allocated RB, it makes
                # no sense to use ACS.
                # At the same time, ignoring ACS doesn't seem correct since the interference
                # could DECREASE when it would make sense for it to increase.
                # e.g. adjacent systems -> slightly co-channel with ACS = inf
                # should interfer ^        less than this ^

                if self.parameters.imt.adjacent_ch_reception == "ACS":
                    if self.overlapping_bandwidth:
                        if getattr(self, "_acs_warned"):
                            warn(
                                "You're trying to use ACS on a partially overlapping band "
                                "with UEs.\n\tVerify the code implements the behavior you expect!!"
                            )
                            self._acs_warned = True
                    non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                    acs_dB = self.parameters.imt.ue.adjacent_ch_selectivity

                    # NOTE: only the power not overlapping is attenuated by ACS
                    # tx_pow_adj_lin = PSD * non_overlap_imt_bw
                    # rx_oob = tx_pow_adj_lin / acs
                    rx_oob[::] = self.param_system.tx_power_density + 10 * np.log10(non_overlap_sys_bw * 1e6) - acs_dB
                elif self.parameters.imt.adjacent_ch_reception == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for parameters.imt.adjacent_ch_reception == {
                            self.parameters.imt.adjacent_ch_reception}")

                # for tx oob we accept ACLR and spectral mask
                if self.param_system.adjacent_ch_emissions == "SPECTRAL_MASK":
                    ue_bws = self.ue.bandwidth[ue]
                    center_freqs = self.ue.center_freq[ue]

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                category=RuntimeWarning,
                                                message="divide by zero encountered in log10")
                        for i, center_freq, bw in zip(
                                range(len(center_freqs)), center_freqs, ue_bws):
                            # calculate tx emissions in UE in use bandwidth only
                            # [dB]
                            tx_oob[i] = self.system.spectral_mask.power_calc(
                                center_freq,
                                bw
                            ) - 30
                elif self.param_system.adjacent_ch_emissions == "ACLR":
                    # consider ACLR only over non co-channel RBs
                    # This should diminish some of the ACLR interference
                    # in a way that make sense
                    non_overlap_imt_bw = self.ue.bandwidth[ue] * (1. - weights)
                    # NOTE: approximated equal to IMT bw
                    measurement_bw = self.param_system.bandwidth
                    aclr_dB = self.param_system.adjacent_ch_leak_ratio

                    if self.parameters.imt.bandwidth - self.overlapping_bandwidth > measurement_bw:
                        # NOTE: ACLR defines total leaked power over a fixed measurement bandwidth.
                        # If the victim bandwidth is wider, you’re assuming the same leakage
                        # profile extends beyond the ACLR-defined region, which may overestimate interference
                        # FIXME: if the victim bw fully contains tx bw, then
                        # EACH region should be <= measurement_bw
                        warn(
                            "Using System ACLR into IMT, but ACLR measurement bw is "
                            f"{measurement_bw} while the IMT bw is bigger ({self.parameters.imt.bandwidth}).\n"
                            "Are you sure you intend to apply the same ACLR to the entire IMT bw?"
                        )

                    # tx_oob_in_measurement = (tx_pow_lin / aclr)
                    # => approx. PSD = (tx_pow_lin / aclr) / measurement_bw
                    # approximated received tx_oob = PSD * non_overlap_imt_bw
                    tx_oob[::] = self.param_system.tx_power_density + \
                        10 * np.log10(1e6) -  \
                        aclr_dB + 10 * np.log10(
                            non_overlap_imt_bw)
                elif self.param_system.adjacent_ch_emissions == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for param_system.adjacent_ch_emissions == {
                            self.param_system.adjacent_ch_emissions}")

                if self.param_system.adjacent_ch_emissions != "OFF":
                    tx_oob = tx_oob[:, np.newaxis] - self.coupling_loss_imt_system[ue, :][:, active_sys]

                rx_oob = rx_oob[:, np.newaxis] - self.coupling_loss_imt_system_adjacent[ue, :][:, active_sys]

                # Out of band power
                # sum linearly power leaked into band and power received in the
                # adjacent band
                oob_power = 10 * np.log10(
                    10 ** (0.1 * tx_oob) + 10 ** (0.1 * rx_oob)
                )
            # Total external interference into the UE in dBm
            ue_ext_int = 10 * np.log10(np.power(10,
                                                0.1 * in_band_interf_power) + np.power(10,
                                                                                       0.1 * oob_power))

            # Sum all the interferers for each UE
            self.ue.ext_interference[ue] = 10 * \
                np.log10(np.sum(np.power(10, 0.1 * ue_ext_int), axis=1)) + 30

            self.ue.sinr_ext[ue] = \
                self.ue.rx_power[ue] - (10 * np.log10(np.power(10, 0.1 * self.ue.total_interference[ue]) +
                                                      np.power(10, 0.1 * (self.ue.ext_interference[ue]))))

            # Calculate INR in dB
            self.ue.thermal_noise[ue] = \
                10 * np.log10(BOLTZMANN_CONSTANT * self.parameters.imt.noise_temperature * 1e3) + \
                10 * np.log10(self.ue.bandwidth[ue] * 1e6) + self.parameters.imt.ue.noise_figure

            self.ue.inr[ue] = self.ue.ext_interference[ue] - \
                self.ue.thermal_noise[ue]

        # Calculate PFD at the UE

        # Distance from each system transmitter to each UE receiver (in meters)
        dist_sys_to_imt = self.system.get_3d_distance_to(
            self.ue)  # shape: [n_tx, n_ue]

        # EIRP in dBW/MHz per transmitter
        eirp_dBW_MHz = self.param_system.tx_power_density + \
            60 + self.system_imt_antenna_gain

        # PFD formula (dBW/m²/MHz)
        # PFD = EIRP - 10log10(4π) - 20log10(distance)
        # Store the PFD for each transmitter and each UE
        self.ue.pfd_external = eirp_dBW_MHz - \
            10.992098640220963 - 20 * np.log10(dist_sys_to_imt)

        # Total PFD per UE (sum of PFDs from each transmitter)
        # Convert PFD from dB to linear scale (W/m²/MHz)
        pfd_linear = 10 ** (self.ue.pfd_external / 10)
        # Sum PFDs from all transmitters for each UE (axis=0 assumes shape
        # [n_tx, n_ue])
        pfd_agg_linear = np.sum(pfd_linear[active_sys], axis=0)
        # Convert back to dBW
        self.ue.pfd_external_aggregated = 10 * np.log10(pfd_agg_linear)

    def calculate_external_interference(self):
        """
        Calculates interference that IMT system generates on other system
        """
        if self.co_channel or (
            self.adjacent_channel and self.param_system.adjacent_ch_reception != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system, self.bs, is_co_channel=True, )
        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.bs,
                    is_co_channel=False,
                )

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the interfered systems bandwidth
        # calculate interference only from active UE's
        rx_interference = 0

        bs_active = np.where(self.bs.active)[0]
        sys_active = np.where(self.system.active)[0]
        for bs in bs_active:
            active_beams = [
                i for i in range(
                    bs *
                    self.parameters.imt.ue.k, (bs + 1) *
                    self.parameters.imt.ue.k,
                )
            ]

            if self.co_channel:
                ue = self.link[bs]
                weights = self.calculate_bw_weights(
                    self.ue.bandwidth[ue],
                    self.ue.center_freq[ue],
                    self.param_system.bandwidth,
                    self.param_system.frequency,
                )

                interference = self.bs.tx_power[bs] - \
                    self.coupling_loss_imt_system[active_beams, sys_active]
                rx_interference += np.sum(
                    weights * np.power(
                        10,
                        0.1 * interference,
                    ),
                )

            if self.adjacent_channel:
                # These are in dB. Turn to zero linear.
                tx_oob = -np.inf
                rx_oob = -np.inf
                # Calculate how much power is emitted in the adjacent channel:
                if self.parameters.imt.adjacent_ch_emissions == "SPECTRAL_MASK":
                    # The unwanted emission is calculated in terms of TRP (after
                    # antenna). In SHARC implementation, ohmic losses are already
                    # included in coupling loss. Then, care has to be taken;
                    # otherwise ohmic loss will be included twice.
                    tx_oob = self.bs.spectral_mask.power_calc(
                        self.param_system.frequency,
                        self.system.bandwidth) + self.parameters.imt.bs.ohmic_loss

                elif self.parameters.imt.adjacent_ch_emissions == "ACLR":
                    non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                    # NOTE: approximated equal to IMT bw
                    measurement_bw = self.parameters.imt.bandwidth
                    aclr_dB = self.parameters.imt.bs.adjacent_ch_leak_ratio

                    if non_overlap_sys_bw > measurement_bw:
                        # NOTE: ACLR defines total leaked power over a fixed measurement bandwidth.
                        # If the victim bandwidth is wider, you’re assuming the same leakage
                        # profile extends beyond the ACLR-defined region, which may overestimate interference
                        # FIXME: if the victim bw fully contains tx bw, then
                        # EACH region should be <= measurement_bw
                        warn(
                            "Using IMT ACLR into system, but ACLR measurement bw is "
                            f"{measurement_bw} while the system bw is bigger ({non_overlap_sys_bw}).\n"
                            "Are you sure you intend to apply ACLR to the entire system bw?"
                        )

                    # tx_oob_in_measurement = (tx_pow_lin / aclr)
                    # => approx. PSD = (tx_pow_lin / aclr) / measurement_bw
                    # approximated received tx_oob = PSD * non_overlap_sys_bw
                    # NOTE: we don't get total power, but power per beam
                    # because later broadcast will sum this tx_oob `k` times
                    tx_oob = self.bs.tx_power[bs] - aclr_dB + 10 * np.log10(
                        non_overlap_sys_bw / measurement_bw
                    )
                elif self.parameters.imt.adjacent_ch_emissions == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for self.parameters.imt.adjacent_ch_emissions == {
                            self.parameters.imt.adjacent_ch_emissions}")

                # Calculate how much power is received in the adjacent channel
                if self.param_system.adjacent_ch_reception == "ACS":
                    non_overlap_imt_bw = self.parameters.imt.bandwidth - self.overlapping_bandwidth
                    tx_bw = self.parameters.imt.bandwidth
                    acs_dB = self.param_system.adjacent_ch_selectivity

                    # NOTE: only the power not overlapping is attenuated by ACS
                    # PSD = tx_pow_lin / tx_bw
                    # tx_pow_adj_lin = PSD * non_overlap_imt_bw
                    # rx_oob = tx_pow_adj_lin / acs
                    rx_oob = self.bs.tx_power[bs] + 10 * np.log10(
                        non_overlap_imt_bw / tx_bw
                    ) - acs_dB
                elif self.param_system.adjacent_ch_reception == "OFF":
                    if self.parameters.imt.adjacent_ch_emissions == "OFF":
                        raise ValueError(
                            "parameters.imt.adjacent_ch_emissions and parameters.imt.adjacent_ch_reception"
                            " cannot be both set to \"OFF\"")
                else:
                    raise ValueError(
                        f"No implementation for self.param_system.adjacent_ch_reception == {
                            self.param_system.adjacent_ch_reception}")

                # oob_power per beam
                # NOTE: we only consider one beam since all beams should have gain
                # of a single element for IMT, and as such the coupling loss should be the
                # same for all beams
                adj_loss = self.coupling_loss_imt_system_adjacent[np.ix_(active_beams, sys_active)]

                # FIXME: for more than 1 sys
                # NOTE: sharc impl already doesn't really work with n_sys > 1
                # so more would have to be fixed before this
                assert np.all(adj_loss == adj_loss.flat[0])

                tx_oob -= adj_loss[0, :]
                if self.param_system.adjacent_ch_reception != "OFF":
                    rx_oob -= self.coupling_loss_imt_system[active_beams, sys_active]

                # Out of band power
                # sum linearly power leaked into band and power received in the
                # adjacent band
                oob_power = 10 * np.log10(
                    10 ** (0.1 * tx_oob) + 10 ** (0.1 * rx_oob)
                )

                # System rx interference
                rx_interference += np.sum(
                    np.power(10, 0.1 * oob_power)
                )

        # Total received interference - dBW
        self.system.rx_interference = 10 * np.log10(rx_interference)
        # calculate N
        self.system.thermal_noise = \
            10 * math.log10(BOLTZMANN_CONSTANT * self.system.noise_temperature * 1e3) + \
            10 * math.log10(self.param_system.bandwidth * 1e6)

        # Calculate INR at the system - dBm
        self.system.inr = np.array(
            [self.system.rx_interference - self.system.thermal_noise],
        )

        # Calculate PFD at the system
        # TODO: generalize this a bit more if needed
        if hasattr(
                self.system.antenna[0],
                "effective_area") and self.system.num_stations == 1:
            self.system.pfd = 10 * \
                np.log10(
                    10**(self.system.rx_interference / 10) /
                    self.system.antenna[0].effective_area,
                )

    def collect_results(self, write_to_file: bool, snapshot_number: int):
        """
        Collect and store results for the current downlink simulation snapshot.

        Args:
            write_to_file (bool): Whether to write results to file.
            snapshot_number (int): The current snapshot number.
        """
        if not self.parameters.imt.interfered_with and np.any(self.bs.active):
            self.results.system_inr.extend(self.system.inr.flatten())
            self.results.system_dl_interf_power.extend(
                self.system.rx_interference.flatten(),
            )
            self.results.system_dl_interf_power_per_mhz.extend(
                self.system.rx_interference.flatten() - 10 * math.log10(self.system.bandwidth),
            )
            # TODO: generalize this a bit more if needed (same conditional as
            # above)
            if hasattr(
                    self.system.antenna[0],
                    "effective_area") and self.system.num_stations == 1:
                self.results.system_pfd.extend([self.system.pfd])

        bs_active = np.where(self.bs.active)[0]
        sys_active = np.where(self.system.active)[0]
        for bs in bs_active:
            ue = self.link[bs]

            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.results.imt_path_loss.extend(self.path_loss_imt[bs, ue])
                self.results.imt_coupling_loss.extend(
                    self.coupling_loss_imt[bs, ue],
                )

                self.results.imt_bs_antenna_gain.extend(
                    self.imt_bs_antenna_gain[bs, ue],
                )
                self.results.imt_ue_antenna_gain.extend(
                    self.imt_ue_antenna_gain[bs, ue],
                )

                tput = self.calculate_imt_tput(
                    self.ue.sinr[ue],
                    self.parameters.imt.downlink.sinr_min,
                    self.parameters.imt.downlink.sinr_max,
                    self.parameters.imt.downlink.attenuation_factor,
                )
                self.results.imt_dl_tput.extend(tput.tolist())

            # Results for IMT-SYSTEM
            if self.parameters.imt.interfered_with:  # IMT suffers interference
                tput_ext = self.calculate_imt_tput(
                    self.ue.sinr_ext[ue],
                    self.parameters.imt.downlink.sinr_min,
                    self.parameters.imt.downlink.sinr_max,
                    self.parameters.imt.downlink.attenuation_factor,
                )
                self.results.imt_dl_tput_ext.extend(tput_ext.tolist())
                self.results.imt_dl_sinr_ext.extend(
                    self.ue.sinr_ext[ue].tolist(),
                )
                self.results.imt_dl_inr.extend(self.ue.inr[ue].tolist())

                self.results.imt_dl_pfd_external.extend(
                    self.ue.pfd_external[sys_active[:, np.newaxis], ue].flatten())

                self.results.imt_dl_pfd_external_aggregated.extend(
                    self.ue.pfd_external_aggregated[ue].tolist())

                self.results.system_imt_antenna_gain.extend(
                    self.system_imt_antenna_gain[sys_active[:, np.newaxis], ue].flatten(),
                )
                if len(self.imt_system_antenna_gain):
                    self.results.imt_system_antenna_gain.extend(
                        self.imt_system_antenna_gain[sys_active[:, np.newaxis], ue].flatten(),
                    )
                if len(self.imt_system_antenna_gain_adjacent):
                    self.results.imt_system_antenna_gain_adjacent.extend(
                        self.imt_system_antenna_gain_adjacent[sys_active[:, np.newaxis], ue].flatten(),
                    )
                self.results.imt_system_path_loss.extend(
                    self.imt_system_path_loss[sys_active[:, np.newaxis], ue].flatten(),
                )
                if self.param_system.channel_model == "HDFSS":
                    self.results.imt_system_build_entry_loss.extend(
                        self.imt_system_build_entry_loss[sys_active[:, np.newaxis], ue].flatten(),
                    )
                    self.results.imt_system_diffraction_loss.extend(
                        self.imt_system_diffraction_loss[sys_active[:, np.newaxis], ue].flatten(),
                    )
                self.results.sys_to_imt_coupling_loss.extend(
                    self.coupling_loss_imt_system[np.array(ue)[:, np.newaxis], sys_active].flatten())
            else:  # IMT is the interferer
                self.results.system_imt_antenna_gain.extend(
                    self.system_imt_antenna_gain[sys_active[:, np.newaxis], ue].flatten(),
                )
                if len(self.imt_system_antenna_gain):
                    self.results.imt_system_antenna_gain.extend(
                        self.imt_system_antenna_gain[sys_active[:, np.newaxis], ue].flatten(),
                    )
                if len(self.imt_system_antenna_gain_adjacent):
                    self.results.imt_system_antenna_gain_adjacent.extend(
                        self.imt_system_antenna_gain_adjacent[sys_active[:, np.newaxis], ue].flatten(),
                    )
                self.results.imt_system_path_loss.extend(
                    self.imt_system_path_loss[sys_active[:, np.newaxis], ue].flatten(),
                )
                if self.param_system.channel_model == "HDFSS":
                    self.results.imt_system_build_entry_loss.extend(
                        self.imt_system_build_entry_loss[:, bs],
                    )
                    self.results.imt_system_diffraction_loss.extend(
                        self.imt_system_diffraction_loss[:, bs],
                    )

            self.results.imt_dl_tx_power.extend(self.bs.tx_power[bs].tolist())

            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.results.imt_dl_sinr.extend(self.ue.sinr[ue].tolist())
                self.results.imt_dl_snr.extend(self.ue.snr[ue].tolist())

        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)
