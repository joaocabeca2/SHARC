import unittest
from sharc.parameters.parameters_mss_d2d import ParametersOrbit, ParametersMssD2d
from sharc.parameters.imt.parameters_imt import ParametersImt
from sharc.topology.topology_single_base_station_spherical import TopologySingleBaseStationSpherical
from sharc.support.enumerations import StationType
from sharc.station_factory import StationFactory
from sharc.station_manager import StationManager

import numpy as np
import numpy.testing as npt


class StationFactoryNgsoTest(unittest.TestCase):
    def setUp(self):
        # Adding multiple shells to this constellation
        # Creating orbital parameters for the first orbit
        orbit_1 = ParametersOrbit(
            n_planes=20,                  # Number of orbital planes
            sats_per_plane=32,            # Satellites per plane
            phasing_deg=3.9,              # Phasing angle in degrees
            long_asc_deg=18.0,            # Longitude of ascending node
            inclination_deg=54.5,         # Orbital inclination in degrees
            perigee_alt_km=525.0,         # Perigee altitude in kilometers
            apogee_alt_km=525.0           # Apogee altitude in kilometers
        )

        # Creating orbital parameters for the second orbit
        orbit_2 = ParametersOrbit(
            n_planes=12,                  # Number of orbital planes
            sats_per_plane=20,            # Satellites per plane
            phasing_deg=2.0,              # Phasing angle in degrees
            long_asc_deg=30.0,            # Longitude of ascending node
            inclination_deg=26.0,         # Orbital inclination in degrees
            perigee_alt_km=580.0,         # Perigee altitude in kilometers
            apogee_alt_km=580.0           # Apogee altitude in kilometers
        )

        # Creating an NGSO constellation and adding the defined orbits
        param = ParametersMssD2d(
            name="Acme-Star-1",                         # Name of the constellation
            antenna_pattern="ITU-R-S.1528-Taylor",     # Antenna type
            antenna_gain=30.0,                 # Maximum antenna gain in dBi
            orbits=[orbit_1, orbit_2]                   # List of orbital parameters
        )

        # Creating an IMT topology
        imt_topology = TopologySingleBaseStationSpherical(
            cell_radius=500,
            num_clusters=2,
            central_latitude=-15.7801,
            central_longitude=-47.9292
        )

        # random number generator
        rng = np.random.RandomState(seed=42)

        self.ngso_manager = StationFactory.generate_mss_d2d(param, rng, imt_topology)

    def test_ngso_manager(self):
        self.assertEqual(self.ngso_manager.station_type, StationType.MSS_D2D)
        self.assertEqual(self.ngso_manager.num_stations, 20 * 32 + 12 * 20)
        self.assertEqual(self.ngso_manager.x.shape, (20 * 32 + 12 * 20,))
        self.assertEqual(self.ngso_manager.y.shape, (20 * 32 + 12 * 20,))
        self.assertEqual(self.ngso_manager.height.shape, (20 * 32 + 12 * 20,))

    def test_satellite_antenna_pointing(self):
        # by default, satellites should always point to nadir (earth center)

        # Test: check if azimuth is pointing towards correct direction
        # y > 0 <=> azimuth < 0
        # y < 0 <=> azimuth > 0
        npt.assert_array_equal(np.sign(self.ngso_manager.azimuth), -np.sign(self.ngso_manager.y))

        # Test: check if elevation is pointing towards correct direction
        # z > 0 <=> elevation < 0
        # z < 0 <=> elevation > 0
        npt.assert_array_equal(np.sign(self.ngso_manager.elevation), -np.sign(self.ngso_manager.z))

        # Test: check if center of earth is 0deg off axis, and that its distance to satellite is correct
        earth_center = StationManager(1)
        earth_center.x = np.array([0.])
        earth_center.y = np.array([0.])
        earth_center.z = np.array([0.])

        off_axis_angle = self.ngso_manager.get_off_axis_angle(earth_center)
        distance_to_center_of_earth = self.ngso_manager.get_3d_distance_to(earth_center)
        distance_to_center_of_earth_should_eq = np.sqrt(
            self.ngso_manager.x ** 2 + self.ngso_manager.y ** 2 + self.ngso_manager.z ** 2
        )

        npt.assert_allclose(off_axis_angle, 0.0, atol=1e-05)

        npt.assert_allclose(
            distance_to_center_of_earth.flatten(),
            distance_to_center_of_earth_should_eq,
            atol=1e-05
        )


if __name__ == '__main__':
    unittest.main()
