# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:10:14 2019

@author: edgar
"""

import unittest
import numpy as np
import numpy.testing as npt
import math

from sharc.simulation_downlink import SimulationDownlink
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.propagation.propagation_factory import PropagationFactory


class SimulationDownlinkTvroTest(unittest.TestCase):

    def setUp(self):
        self.param = Parameters()

        self.param.general.imt_link = "DOWNLINK"
        self.param.system = "FSS_ES"
        self.param.general.enable_cochannel = True
        self.param.general.enable_adjacent_channel = False
        self.param.general.seed = 101
        self.param.general.overwrite_output = True

        self.param.imt.topology.type = "SINGLE_BS"
        self.param.imt.topology.single_bs.num_clusters = 2
        self.param.imt.topology.single_bs.intersite_distance = 150
        self.param.imt.topology.single_bs.cell_radius = 100
        self.param.imt.minimum_separation_distance_bs_ue = 10
        self.param.imt.interfered_with = False
        self.param.imt.frequency = 3590.0
        self.param.imt.bandwidth = 20
        self.param.imt.rb_bandwidth = 0.180
        self.param.imt.spectral_mask = "3GPP E-UTRA"
        self.param.imt.spurious_emissions = -13
        self.param.imt.guard_band_ratio = 0.1
        self.param.imt.bs.load_probability = 1

        self.param.imt.bs.conducted_power = 46
        self.param.imt.bs.height = 20
        self.param.imt.bs.noise_figure = 5
        self.param.imt.bs.ohmic_loss = 3
        self.param.imt.uplink.attenuation_factor = 0.4
        self.param.imt.uplink.sinr_min = -10
        self.param.imt.uplink.sinr_max = 22
        self.param.imt.ue.k = 2
        self.param.imt.ue.k_m = 1
        self.param.imt.ue.indoor_percent = 0
        self.param.imt.ue.distribution_type = "ANGLE_AND_DISTANCE"
        self.param.imt.ue.distribution_distance = "UNIFORM"
        self.param.imt.ue.distribution_azimuth = "UNIFORM"
        self.param.imt.ue.tx_power_control = "OFF"
        self.param.imt.ue.p_o_pusch = -95
        self.param.imt.ue.alpha = 1
        self.param.imt.ue.p_cmax = 23
        self.param.imt.ue.power_dynamic_range = 63
        self.param.imt.ue.height = 1.5
        self.param.imt.ue.acs = 25
        self.param.imt.ue.noise_figure = 9
        self.param.imt.ue.ohmic_loss = 3
        self.param.imt.ue.body_loss = 4
        self.param.imt.downlink.attenuation_factor = 0.6
        self.param.imt.downlink.sinr_min = -10
        self.param.imt.downlink.sinr_max = 30
        self.param.imt.channel_model = "FSPL"
        self.param.imt.los_adjustment_factor = 29
        # probability of line-of-sight (not for FSPL)
        self.param.imt.line_of_sight_prob = 0.75
        self.param.imt.shadowing = False
        self.param.imt.noise_temperature = 290

        self.param.imt.bs.antenna.adjacent_antenna_model = "BEAMFORMING"
        self.param.imt.ue.antenna.adjacent_antenna_model = "BEAMFORMING"
        self.param.imt.bs.antenna.normalization = False
        self.param.imt.bs.antenna.element_pattern = "F1336"
        self.param.imt.bs.antenna.normalization_file = None
        self.param.imt.bs.antenna.minimum_array_gain = -200
        self.param.imt.bs.antenna.element_max_g = 18
        self.param.imt.bs.antenna.element_phi_3db = 65
        self.param.imt.bs.antenna.element_theta_3db = 0
        self.param.imt.bs.antenna.element_am = 25
        self.param.imt.bs.antenna.element_sla_v = 25
        self.param.imt.bs.antenna.n_rows = 1
        self.param.imt.bs.antenna.n_columns = 1
        self.param.imt.bs.antenna.element_horiz_spacing = 1
        self.param.imt.bs.antenna.element_vert_spacing = 1
        self.param.imt.bs.antenna.multiplication_factor = 12
        self.param.imt.bs.antenna.downtilt = 10

        self.param.imt.ue.antenna.element_pattern = "FIXED"
        self.param.imt.ue.antenna.normalization = False
        self.param.imt.ue.antenna.normalization_file = None
        self.param.imt.ue.antenna.minimum_array_gain = -200
        self.param.imt.ue.antenna.element_max_g = -4
        self.param.imt.ue.antenna.element_phi_3db = 0
        self.param.imt.ue.antenna.element_theta_3db = 0
        self.param.imt.ue.antenna.element_am = 0
        self.param.imt.ue.antenna.element_sla_v = 0
        self.param.imt.ue.antenna.n_rows = 1
        self.param.imt.ue.antenna.n_columns = 1
        self.param.imt.ue.antenna.element_horiz_spacing = 0.5
        self.param.imt.ue.antenna.element_vert_spacing = 0.5
        self.param.imt.ue.antenna.multiplication_factor = 12

        self.param.fss_es.location = "FIXED"
        self.param.fss_es.x = 100
        self.param.fss_es.y = 0
        self.param.fss_es.min_dist_to_bs = 10
        self.param.fss_es.max_dist_to_bs = 600
        self.param.fss_es.height = 6
        self.param.fss_es.elevation_min = 49.8
        self.param.fss_es.elevation_max = 49.8
        self.param.fss_es.azimuth = "180"
        self.param.fss_es.frequency = 3628
        self.param.fss_es.bandwidth = 6
        self.param.fss_es.noise_temperature = 100
        self.param.fss_es.adjacent_ch_selectivity = 0
        self.param.fss_es.tx_power_density = -60
        self.param.fss_es.antenna_gain = 32
        self.param.fss_es.antenna_pattern = "Modified ITU-R S.465"
        self.param.fss_es.diameter = 1.8
        self.param.fss_es.antenna_envelope_gain = 0
        self.param.fss_es.channel_model = "FSPL"
        self.param.fss_es.line_of_sight_prob = 1

    def test_simulation_1bs_1ue_tvro(self):
        self.param.general.system = "FSS_ES"

        self.simulation = SimulationDownlink(self.param, "")
        self.simulation.initialize()
        random_number_gen = np.random.RandomState(self.param.general.seed)

        self.assertTrue(self.simulation.co_channel)

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.imt.bs.antenna,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.x = np.array([0, -200])
        self.simulation.bs.y = np.array([0, 0])
        self.simulation.bs.azimuth = np.array([0, 180])
        self.simulation.bs.elevation = np.array([-10, -10])

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.imt.ue.antenna,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([30, 60, -220, -300])
        self.simulation.ue.y = np.array([0, 0, 0, 0])

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.simulation.select_ue(random_number_gen)
        self.simulation.link = {0: [0, 1], 1: [2, 3]}
        self.assertEqual(self.simulation.link, {0: [0, 1], 1: [2, 3]})

        self.simulation.propagation_imt = PropagationFactory.create_propagation(self.param.imt.channel_model,
                                                                                self.param,
                                                                                self.simulation.param_system,
                                                                                random_number_gen)
        self.simulation.propagation_system = PropagationFactory.create_propagation(self.param.fss_es.channel_model,
                                                                                   self.simulation.param_system,
                                                                                   self.param, random_number_gen)

        # test coupling loss method
        self.simulation.coupling_loss_imt = self.simulation.calculate_intra_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.bs)
        path_loss_imt = np.array([[74.49, 79.50, 90.43, 93.11],
                                  [90.81, 91.87, 72.25, 83.69]])
        bs_antenna_gains = np.array([[3.04, 7.30, -6.45, -6.45],
                                     [-6.45, -6.45, 1.63, 17.95]])
        ue_antenna_gains = np.array([[-4, -4, -4, -4], [-4, -4, -4, -4]])
        coupling_loss_imt = path_loss_imt - bs_antenna_gains - ue_antenna_gains \
            + self.param.imt.bs.ohmic_loss \
            + self.param.imt.ue.ohmic_loss \
            + self.param.imt.ue.body_loss
        npt.assert_allclose(self.simulation.coupling_loss_imt,
                            coupling_loss_imt,
                            atol=1e-1)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 20 / 2)
        npt.assert_allclose(self.simulation.ue.bandwidth,
                            bandwidth_per_ue * np.ones(4), atol=1e-2)

        # there is no power control, so BS's will transmit at maximum power
        self.simulation.power_control()
        tx_power = 46 - 10 * np.log10(2)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array(
            [tx_power, tx_power]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array(
            [tx_power, tx_power]), atol=1e-2)

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        # check UE received power
        rx_power = tx_power - \
            np.concatenate(
                (coupling_loss_imt[0][:2], coupling_loss_imt[1][2:]))
        npt.assert_allclose(self.simulation.ue.rx_power, rx_power, atol=1e-1)

        # check UE received interference
        rx_interference = tx_power - \
            np.concatenate(
                (coupling_loss_imt[1][:2], coupling_loss_imt[0][2:]))
        npt.assert_allclose(self.simulation.ue.rx_interference,
                            rx_interference, atol=1e-1)

        # check UE thermal noise
        thermal_noise = 10 * \
            np.log10(1.38064852e-23 * 290 * bandwidth_per_ue * 1e3 * 1e6) + 9
        npt.assert_allclose(self.simulation.ue.thermal_noise,
                            thermal_noise, atol=1e-1)

        # check UE thermal noise + interference
        total_interference = 10 * \
            np.log10(np.power(10, 0.1 * rx_interference) +
                     np.power(10, 0.1 * thermal_noise))
        npt.assert_allclose(
            self.simulation.ue.total_interference, total_interference, atol=1e-1)

        # check SNR
        npt.assert_allclose(self.simulation.ue.snr,
                            rx_power - thermal_noise, atol=1e-1)

        # check SINR
        npt.assert_allclose(self.simulation.ue.sinr,
                            rx_power - total_interference, atol=1e-1)

        #######################################################################
        self.simulation.system = StationFactory.generate_fss_earth_station(
            self.param.fss_es, random_number_gen)
        self.simulation.system.x = np.array([600])
        self.simulation.system.y = np.array([0])

        # test the method that calculates interference from IMT UE to FSS space
        # station
        self.simulation.calculate_external_interference()

        # check coupling loss from IMT_BS to FSS_ES
        # 4 values because we have 2 BS * 2 beams for each base station.
        path_loss_imt_system = np.array([99.11, 99.11, 101.61, 101.61])
        polarization_loss = 3
        tvro_antenna_gain = np.array([0, 0, 0, 0])
        bs_antenna_gain = np.array([6.47, 6.47, -6.45, -6.45])
        coupling_loss_imt_system = path_loss_imt_system - tvro_antenna_gain \
            - bs_antenna_gain \
            + polarization_loss \
            + self.param.imt.bs.ohmic_loss

        npt.assert_allclose(self.simulation.coupling_loss_imt_system,
                            coupling_loss_imt_system,
                            atol=1e-1)

        # check blocking signal
        interference = tx_power - coupling_loss_imt_system
        rx_interference = 10 * \
            math.log10(np.sum(np.power(10, 0.1 * interference)))
        self.assertAlmostEqual(self.simulation.system.rx_interference,
                               rx_interference,
                               delta=.01)


if __name__ == '__main__':
    unittest.main()
