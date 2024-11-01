# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:43:06 2018

@author: Calil
"""


import unittest
import numpy as np
import numpy.testing as npt
import os.path as path

from sharc.simulation_downlink import SimulationDownlink
from sharc.parameters.parameters import Parameters
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.station_factory import StationFactory
from sharc.parameters.imt.parameters_imt_topology import ParametersImtTopology
from sharc.parameters.imt.parameters_indoor import ParametersIndoor


class SimulationIndoorTest(unittest.TestCase):

    def setUp(self):
        self.param = Parameters()

        self.param.general.imt_link = "DOWNLINK"
        self.param.general.seed = 101
        self.param.general.enable_cochannel = True
        self.param.general.enable_adjacent_channel = False
        self.param.general.overwrite_output = True

        self.param.imt.topology.type = "INDOOR"
        self.param.imt.minimum_separation_distance_bs_ue = 10
        self.param.imt.interfered_with = False
        self.param.imt.frequency = 40000
        self.param.imt.bandwidth = 200
        self.param.imt.rb_bandwidth = 0.180
        self.param.imt.spectral_mask = "IMT-2020"
        self.param.imt.spurious_emissions = -13
        self.param.imt.guard_band_ratio = 0.1
        self.param.imt.bs.load_probability = 1

        self.param.imt.bs.conducted_power = 2
        self.param.imt.bs.height = 3
        self.param.imt.bs.noise_figure = 12
        self.param.imt.bs.ohmic_loss = 3
        self.param.imt.uplink.attenuation_factor = 0.4
        self.param.imt.uplink.sinr_min = -10
        self.param.imt.uplink.sinr_max = 22
        self.param.imt.ue.k = 1
        self.param.imt.ue.k_m = 1
        self.param.imt.ue.indoor_percent = 95
        self.param.imt.ue.distribution_type = "ANGLE_AND_DISTANCE"
        self.param.imt.ue.distribution_distance = "RAYLEIGH"
        self.param.imt.ue.distribution_azimuth = "UNIFORM"
        self.param.imt.ue.tx_power_control = "OFF"
        self.param.imt.ue.p_o_pusch = -95
        self.param.imt.ue.alpha = 1
        self.param.imt.ue.p_cmax = 22
        self.param.imt.ue.height = 1.5
        self.param.imt.ue.noise_figure = 12
        self.param.imt.ue.ohmic_loss = 3
        self.param.imt.ue.body_loss = 4
        self.param.imt.downlink.attenuation_factor = 0.6
        self.param.imt.downlink.sinr_min = -10
        self.param.imt.downlink.sinr_max = 30
        self.param.imt.channel_model = "FSPL"
        self.param.imt.shadowing = False
        self.param.imt.noise_temperature = 290

        self.param.imt.bs.antenna.adjacent_antenna_model = "SINGLE_ELEMENT"
        self.param.imt.ue.antenna.adjacent_antenna_model = "SINGLE_ELEMENT"
        self.param.imt.bs.antenna.normalization = False
        self.param.imt.bs.antenna.normalization_file = path.join(
            '..', 'sharc', 'antenna', 'beamforming_normalization', 'bs_indoor_norm.npz')
        self.param.imt.ue.antenna.normalization_file = path.join(
            '..', 'sharc', 'antenna', 'beamforming_normalization', 'ue_norm.npz')
        self.param.imt.bs.antenna.element_pattern = "M2101"
        self.param.imt.bs.antenna.minimum_array_gain = -200
        self.param.imt.bs.antenna.element_max_g = 5
        self.param.imt.bs.antenna.element_phi_3db = 90
        self.param.imt.bs.antenna.element_theta_3db = 90
        self.param.imt.bs.antenna.element_am = 25
        self.param.imt.bs.antenna.element_sla_v = 25
        self.param.imt.bs.antenna.n_rows = 8
        self.param.imt.bs.antenna.n_columns = 16
        self.param.imt.bs.antenna.element_horiz_spacing = 0.5
        self.param.imt.bs.antenna.element_vert_spacing = 0.5
        self.param.imt.bs.antenna.multiplication_factor = 12
        self.param.imt.bs.antenna.downtilt = 90

        self.param.imt.ue.antenna.element_pattern = "M2101"
        self.param.imt.ue.antenna.normalization = False
        self.param.imt.ue.antenna.minimum_array_gain = -200
        self.param.imt.ue.antenna.element_max_g = 5
        self.param.imt.ue.antenna.element_phi_3db = 90
        self.param.imt.ue.antenna.element_theta_3db = 90
        self.param.imt.ue.antenna.element_am = 25
        self.param.imt.ue.antenna.element_sla_v = 25
        self.param.imt.ue.antenna.n_rows = 4
        self.param.imt.ue.antenna.n_columns = 4
        self.param.imt.ue.antenna.element_horiz_spacing = 0.5
        self.param.imt.ue.antenna.element_vert_spacing = 0.5
        self.param.imt.ue.antenna.multiplication_factor = 12

        self.param.imt.topology.indoor.basic_path_loss = "FSPL"
        self.param.imt.topology.indoor.n_rows = 1
        self.param.imt.topology.indoor.n_colums = 1
        self.param.imt.topology.indoor.num_imt_buildings = 'ALL'
        self.param.imt.topology.indoor.street_width = 30
        self.param.imt.topology.indoor.ue_indoor_percent = 0.95
        self.param.imt.topology.indoor.building_class = "TRADITIONAL"
        self.param.imt.topology.indoor.intersite_distance = 30
        self.param.imt.topology.indoor.num_cells = 4
        self.param.imt.topology.indoor.num_floors = 1

        self.param.fss_es.x = 135
        self.param.fss_es.y = 65
        self.param.fss_es.location = "FIXED"
        self.param.fss_es.height = 10
        self.param.fss_es.elevation_min = 10
        self.param.fss_es.elevation_max = 10
        self.param.fss_es.azimuth = "-180"
        self.param.fss_es.frequency = 40000
        self.param.fss_es.bandwidth = 180
        self.param.fss_es.noise_temperature = 400
        self.param.fss_es.tx_power_density = -69
        self.param.fss_es.antenna_gain = 47
        self.param.fss_es.antenna_pattern = "ITU-R S.580"
        self.param.fss_es.channel_model = "FSPL"
        self.param.fss_es.line_of_sight_prob = 1
        self.param.fss_es.adjacent_ch_selectivity = 0
        self.param.fss_es.diameter = 0.74

    def test_simulation_fss_es(self):
        # Initialize stations
        self.param.general.system = "FSS_ES"

        self.simulation = SimulationDownlink(self.param, "")
        self.simulation.initialize()

        random_number_gen = np.random.RandomState(101)

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.imt.bs.antenna,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.assertTrue(np.all(self.simulation.bs.active))

        self.simulation.system = StationFactory.generate_fss_earth_station(self.param.fss_es,
                                                                           random_number_gen)

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.imt.ue.antenna,
                                                            self.simulation.topology,
                                                            random_number_gen)

#        print("Random position:")
#        self.simulation.plot_scenario()
        self.simulation.ue.x = np.array([0.0, 45.0, 75.0, 120.0])
        self.simulation.ue.y = np.array([0.0, 50.0, 0.0, 50.0])
#        print("Forced position:")
#        self.simulation.plot_scenario()

        # Connect and select UEs
        self.simulation.connect_ue_to_bs()
        self.simulation.select_ue(random_number_gen)
        self.assertTrue(np.all(self.simulation.ue.active))
        self.assertDictEqual(
            self.simulation.link, {
                0: [0], 1: [1], 2: [2], 3: [3]})

        # Test BS-to-UE angles in the IMT coord system
        expected_azi = np.array([[-120.96, 39.80, -22.62, 13.39],
                                 [-150.95, 90.00, -39.81, 18.43],
                                 [-161.57, 140.19, -90.00, 29.06],
                                 [-166.61, 157.38, -140.19, 59.03]])
        npt.assert_allclose(self.simulation.bs_to_ue_phi,
                            expected_azi,
                            atol=1e-2)
        expected_ele = np.array([[92.95, 92.20, 91.32, 90.79],
                                 [91.67, 93.43, 92.20, 91.09],
                                 [91.09, 92.20, 93.43, 91.67],
                                 [90.79, 91.32, 92.20, 92.95]])
        npt.assert_allclose(self.simulation.bs_to_ue_theta,
                            expected_ele,
                            atol=1e-2)

        # Test BS-to-UE angles in the local coord system
        expected_loc = [(np.array([-86.57]), np.array([120.92])),
                        (np.array([86.57]), np.array([90.00])),
                        (np.array([-86.57]), np.array([90.00])),
                        (np.array([86.57]), np.array([59.08]))]
        expected_beam = [(-86.57, 30.92),
                         (86.57, 0.00),
                         (-86.57, 0.00),
                         (86.57, -30.92)]
        for k in range(self.simulation.bs.num_stations):

            self.assertEqual(self.simulation.bs.antenna[k].azimuth, 0.0)
            self.assertEqual(self.simulation.bs.antenna[k].elevation, -90.0)

            lo_angles = self.simulation.bs.antenna[k].to_local_coord(expected_azi[k, k],
                                                                     expected_ele[k, k])
            npt.assert_array_almost_equal(
                lo_angles, expected_loc[k], decimal=2)
            npt.assert_array_almost_equal(self.simulation.bs.antenna[k].beams_list[0],
                                          expected_beam[k], decimal=2)

        # Test angle to ES in the IMT coord system
        phi_es, theta_es = self.simulation.bs.get_pointing_vector_to(
            self.simulation.system)
        expected_phi_es = np.array([[18.44], [23.96], [33.69], [53.13]])
        npt.assert_array_almost_equal(phi_es, expected_phi_es, decimal=2)
        expected_theta_es = np.array([[86.83], [85.94], [84.46], [82.03]])
        npt.assert_array_almost_equal(theta_es, expected_theta_es, decimal=2)

        # Test angle to ES in the local coord system
        expected_es_loc = [(np.array([99.92]), np.array([18.70])),
                           (np.array([99.92]), np.array([24.28])),
                           (np.array([99.92]), np.array([34.09])),
                           (np.array([99.92]), np.array([53.54]))]
        for k in range(self.simulation.bs.num_stations):
            lo_angles = self.simulation.bs.antenna[k].to_local_coord(expected_phi_es[k],
                                                                     expected_theta_es[k])
            npt.assert_array_almost_equal(
                lo_angles, expected_es_loc[k], decimal=2)

        # Test gain to ES
        calc_gain = self.simulation.calculate_gains(self.simulation.bs,
                                                    self.simulation.system)
        for k in range(self.simulation.bs.num_stations):
            beam = 0
            exp_gain = self.simulation.bs.antenna[k]._beam_gain(expected_es_loc[k][0],
                                                                expected_es_loc[k][1],
                                                                beam)
            self.assertAlmostEqual(
                np.ndarray.item(
                    calc_gain[k]),
                np.ndarray.item(exp_gain),
                places=1)


if __name__ == '__main__':
    unittest.main()
