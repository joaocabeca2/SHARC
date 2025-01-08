import os
from dataclasses import dataclass, field
from typing import List
from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit
from sharc.parameters.antenna.parameters_antenna_s1528 import ParametersAntennaS1528

@dataclass
class ParametersNgsoConstellation(ParametersBase):
    """Define parameters for a NGSO Constellation."""
    section_name: str = "ngso"
    name: str = "Default"
    orbits: List[ParametersOrbit] = field(default_factory=list)
    antenna: str = None
    max_transmit_power_dB: float = 46.0
    max_transmit_gain_dBi: float = 30.0
    max_receive_gain_dBi: float = 30.0
    max_num_of_beams: int = 19
    cell_radius_m: float = 19000.0
    antenna: ParametersAntennaS1528 = field(default_factory=ParametersAntennaS1528)

    def load_parameters_from_file(self, config_file: str):
        """Load parameters from file and validate."""
        super().load_parameters_from_file(config_file)

        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"ParametersNgsoConstellation: Invalid name = {self.name}. \
                             Must be a non-empty string.")
        if self.antenna not in ["Taylor1.4", "ITU-R M.1466", "OMNI"]:
            raise ValueError(f"ParametersNgsoConstellation: Invalid antenna = {self.antenna}. \
                             Allowed values are \"Taylor1.4\", \"ITU-R M.1466\", \"OMNI\".")
        if self.max_transmit_gain_dBi < 0:
            raise ValueError(f"ParametersNgsoConstellation: Invalid max_gain_dbi = {self.max_transmit_gain_dBi}. \
                             Gain must be non-negative.")
        if not self.orbits:
            raise ValueError("ParametersNgsoConstellation: No orbits defined. \
                             At least one orbit must be specified.")
        for orbit in self.orbits:
            if not isinstance(orbit, ParametersOrbit):
                raise ValueError("ParametersNgsoConstellation: Invalid orbit configuration. \
                                 All orbits must be instances of ParametersOrbit.")


# Test block
if __name__ == "__main__":
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

    # # Creating an NGSO constellation and adding the defined orbits
    # constellation = ParametersNgsoConstellation(
    #     name="Acme-Star-1",           # Name of the constellation
    #     antenna="Taylor1.4",          # Antenna type
    #     max_transmit_gain_dBi=30.0,            # Maximum antenna ga.in in dBi
    #     orbits=[orbit_1, orbit_2]     # List of orbital parameters
    # )

    # Testing parameter validation

    try:
        constellation = ParametersNgsoConstellation()
        # Load parameters from a configuration file (yaml in this case)
        yaml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../input/ngso_parameters.yaml")
        constellation.load_parameters_from_file(yaml_file_path)
        print("Parameters loaded successfully.")

        # Print constellation details
        print(f"Constellation Name: {constellation.name}")
        print(f"Antenna: {constellation.antenna}")
        print(f"Max Gain (dBi): {constellation.max_transmit_gain_dBi}")
        print("\nLoaded Orbits:")
        for idx, orbit in enumerate(constellation.orbits, start=1):
            print(f"  Orbit {idx}:")
            print(f"    Planes: {orbit.n_planes}")
            print(f"    Satellites per Plane: {orbit.sats_per_plane}")
            print(f"    Inclination (deg): {orbit.inclination_deg}")
            print(f"    Perigee Altitude (km): {orbit.perigee_alt_km}")
            print(f"    Apogee Altitude (km): {orbit.apogee_alt_km}")
            print(f"    Phasing (deg): {orbit.phasing_deg}")
            print(f"    Longitude of Ascending Node (deg): {orbit.long_asc_deg}")
    except ValueError as e:
        print(f"Validation Error: {e}")
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}")
