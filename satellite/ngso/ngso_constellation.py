"""Implementation and definitions of the NGSO Satellite Constellation
"""
from types import List
from sharc.antenna.antenna_s1528 import AntennaS1528

from sharc.station_manager import StationManager

class SpaceStationManager(StationManager):
    def __init__(self, n):
        super().__init__(n)
        self.mean_anomaly = list

    def update_station_positions(sefl):
        pass

    def generate_space_stations(self):

class NgsoOrbit():
    """This class implements the NGSO Constellation according to S.1503."""

    def __init__(self, orbit_params) -> None:
        """NGSO Constellation implementation."""
        self.n_shells = n_shells  # number of orbital shells
        self.Nsp = 6  # number of satellites in the orbital plane (A.4.b.4.b)
        self.Np = 8  # number of orbital planes (A.4.b.2)
        self.phasing = 7.5  # satellite phasing between planes, in degrees
        self.long_asc = 0  # initial longitude of ascending node of the first plane, in degrees (A.4.b.4.j)
        self.omega = 0  # argument of perigee, in degrees (A.4.b.4.i)
        self.delta = 52  # orbital plane inclination, in degrees (A.4.b.4.a)
        self.hp = 1414  # altitude of perigee in km (A.4.b.4.e)
        self.ha = 1414  # altitude of apogee in km (A.4.b.4.d)
        self.Mo = 0  # initial mean anomaly for first satellite of first plane, in degrees
        self.space_stations = StationManager

    def initilize(self):
        # Fill StationManager
        self.space_stations = self.generate_space_stations()

    def generate_space_stations(self):
        # station factory code
        pass

    def update_space_stations_positions(self):
        # update geometry
        self.space_stations.update_station_positions()


class NgsoConstellation():
    def __init__(self, constelation_params) -> None:
        self.name = constelation_params.name
        self.orbits = List[NgsoOrbit]
        for orbit_params in constelation_params.orbits:
            self.orbits.append(NgsoOrbit(orbit_params))

    def initialize(self):
        for orbit in self.orbits:
            orbit.initialize()
    
    def calculate_orbits(self, time_step_sec=5, num_orbital_periods=4):
        # orbit predictor
        pass

    def update_orbits(self):
        for orbit in self.orbits:
            orbit.update_space_stations_positions(self)
        # update orbits at each time step

