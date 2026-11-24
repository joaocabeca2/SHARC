# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:49:09 2017

@author: edgar
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.parameters.imt.parameters_imt import ParametersImt
from sharc.station_factory import StationFactory
from sharc.topology.topology_ntn import TopologyNTN
from sharc.topology.topology_macrocell import TopologyMacrocell
from sharc.parameters.parameters_single_space_station import ParametersSingleSpaceStation


class StationFactoryTest(unittest.TestCase):
    """Test cases for StationFactory."""

    def setUp(self):
        """Set up a StationFactory instance for testing."""
        self.station_factory = StationFactory()

    def test_generate_imt_base_stations(self):
        """Test IMT base station generation (placeholder)."""

    def test_generate_imt_base_stations_ntn(self):
        """Test for IMT-NTN space station generation."""
        # seed = 100  # Unused variable removed
        rng = np.random.RandomState(100)

        param_imt = ParametersImt()
        param_imt.topology.type = "NTN"

        # Paramters for IMT-NTN
        param_imt.topology.ntn.bs_height = 1200000  # meters
        param_imt.topology.ntn.cell_radius = 45000  # meters
        param_imt.topology.ntn.bs_azimuth = 60  # degrees
        param_imt.topology.ntn.bs_elevation = 45  # degrees
        param_imt.topology.ntn.num_sectors = 1

        ntn_topology = TopologyNTN(
            param_imt.topology.ntn.intersite_distance,
            param_imt.topology.ntn.cell_radius,
            param_imt.topology.ntn.bs_height,
            param_imt.topology.ntn.bs_azimuth,
            param_imt.topology.ntn.bs_elevation,
            param_imt.topology.ntn.num_sectors,
        )

        ntn_topology.calculate_coordinates()
        ntn_bs = StationFactory.generate_imt_base_stations(
            param_imt, param_imt.bs.antenna.array, ntn_topology, rng)
        npt.assert_equal(ntn_bs.height, param_imt.topology.ntn.bs_height)
        # the azimuth seen from BS antenna
        npt.assert_almost_equal(
            ntn_bs.azimuth[0],
            param_imt.topology.ntn.bs_azimuth - 180,
            1e-3)
        # Elevation w.r.t to xy plane
        npt.assert_almost_equal(ntn_bs.elevation[0], -45.0, 1e-2)
        npt.assert_almost_equal(
            ntn_bs.x, param_imt.topology.ntn.bs_height *
            np.tan(np.radians(param_imt.topology.ntn.bs_elevation)) *
            np.cos(np.radians(param_imt.topology.ntn.bs_azimuth)), 1e-2,
        )

    def test_generate_imt_ue_outdoor_ntn(self):
        """Basic test for IMT UE NTN generation."""
        seed = 100
        rng = np.random.RandomState(seed)

        # Parameters used for IMT-NTN and UE distribution
        param_imt = ParametersImt()
        param_imt.topology.type = "NTN"
        param_imt.ue.azimuth_range = (-180, 180)
        param_imt.ue.distribution_type = "ANGLE_AND_DISTANCE"
        param_imt.ue.distribution_azimuth = "UNIFORM"
        param_imt.ue.distribution_distance = "UNIFORM"
        param_imt.ue.k = 1000

        # Paramters for IMT-NTN
        param_imt.topology.ntn.bs_height = 1200000  # meters
        param_imt.topology.ntn.cell_radius = 45000  # meters
        param_imt.topology.ntn.bs_azimuth = 60  # degrees
        param_imt.topology.ntn.bs_elevation = 45  # degrees
        param_imt.topology.ntn.num_sectors = 1

        ntn_topology = TopologyNTN(
            param_imt.topology.ntn.intersite_distance,
            param_imt.topology.ntn.cell_radius,
            param_imt.topology.ntn.bs_height,
            param_imt.topology.ntn.bs_azimuth,
            param_imt.topology.ntn.bs_elevation,
            param_imt.topology.ntn.num_sectors,
        )

        ntn_topology.calculate_coordinates()
        ntn_ue = StationFactory.generate_imt_ue_outdoor(
            param_imt, param_imt.ue.antenna.array, rng, ntn_topology)
        dist = np.sqrt(ntn_ue.x**2 + ntn_ue.y**2)
        # test if the maximum distance is close to the cell radius within a
        # 100km range
        npt.assert_almost_equal(
            dist.max(), param_imt.topology.ntn.cell_radius, -2)

    def test_generate_single_space_station(self):
        """Basic test for space station generation."""

        param = ParametersSingleSpaceStation()
        # just passing required parameters:
        param.frequency = 8000
        param.bandwidth = 100
        param.channel_model = "P619"
        param.tx_power_density = -200
        param.geometry.es_altitude = 0
        param.geometry.azimuth.fixed = 0
        param.antenna.pattern = "OMNI"
        param.antenna.gain = 10

        param.geometry.location.type = "FIXED"
        param.geometry.altitude = 35786000.0
        param.geometry.es_lat_deg = 0
        param.geometry.es_long_deg = 0
        param.geometry.location.fixed.lat_deg = 0
        param.geometry.location.fixed.long_deg = 0

        param.propagate_parameters()
        # This should not error on this test:
        param.validate()

        # experimental from simulator
        max_gso_fov = 81.30784

        def get_ground_elevation(ss):
            return np.rad2deg(
                np.arctan2(
                    ss.height,
                    np.sqrt(
                        ss.x**2 +
                        ss.y**2)))

        space_station = StationFactory.generate_single_space_station(param)

        # test if the maximum distance is close to the cell radius within a
        # 100km range
        npt.assert_almost_equal(space_station.height, param.geometry.altitude)
        npt.assert_almost_equal(get_ground_elevation(space_station), 90)

        param.geometry.es_lat_deg = max_gso_fov

        space_station = StationFactory.generate_single_space_station(param)

        npt.assert_almost_equal(get_ground_elevation(space_station), 0, 5)
        npt.assert_almost_equal(space_station.height, 0, 0)

        param.geometry.es_lat_deg = 0
        param.geometry.es_long_deg = max_gso_fov

        space_station = StationFactory.generate_single_space_station(param)

        npt.assert_almost_equal(get_ground_elevation(space_station), 0, 5)
        npt.assert_almost_equal(space_station.height, 0, 0)

        param.geometry.es_long_deg = 0
        param.geometry.location.fixed.lat_deg = max_gso_fov

        space_station = StationFactory.generate_single_space_station(param)

        npt.assert_almost_equal(get_ground_elevation(space_station), 0, 5)
        npt.assert_almost_equal(space_station.height, 0, 0)

        param.geometry.location.fixed.lat_deg = 0
        param.geometry.location.fixed.long_deg = max_gso_fov

        space_station = StationFactory.generate_single_space_station(param)
        npt.assert_almost_equal(get_ground_elevation(space_station), 0, 5)
        npt.assert_almost_equal(space_station.height, 0, 0)

    def test_get_random_position_inside_sector(self):
        """UEs sorteadas por célula devem cair no setor da BS servidora:
        hexágono de lado ISD/3 com a BS num vértice."""
        isd = 500
        topology = TopologyMacrocell(isd, 1)
        topology.calculate_coordinates()
        n_per_cell = 200
        rng = np.random.RandomState(7)
        x, y, _, theta, distance = StationFactory.get_random_position(
            n_per_cell * topology.num_base_stations, topology, rng,
            deterministic_cell=True,
        )
        x, y = np.array(x), np.array(y)
        cell = np.repeat(np.arange(topology.num_base_stations), n_per_cell)

        # Distância máxima ao vértice oposto do setor: 2*ISD/3
        self.assertLessEqual(np.max(distance), 2 * isd / 3 + 1e-6)

        # Dentro do hexágono do setor (centro a ISD/3 na direção do azimute)
        side = isd / 3
        az = np.radians(topology.azimuth[cell])
        cx = topology.x[cell] + side * np.cos(az)
        cy = topology.y[cell] + side * np.sin(az)
        ang = np.arctan2(y - cy, x - cx)
        # distância do centro até a borda de um hexágono com vértices a 0°, 60°, ...
        rel = np.mod(ang, np.pi / 3) - np.pi / 6
        edge = side * np.cos(np.pi / 6) / np.cos(rel)
        self.assertTrue(np.all(np.hypot(x - cx, y - cy) <= edge + 1e-6))


if __name__ == '__main__':
    unittest.main()
