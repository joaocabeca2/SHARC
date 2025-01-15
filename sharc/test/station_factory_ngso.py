from sharc.parameters.parameters_ngso_constellation import ParametersOrbit, ParametersNgsoConstellation
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.station_factory import StationFactory
import numpy as np

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
param = ParametersNgsoConstellation(
    name="Acme-Star-1",           # Name of the constellation
    antenna="Taylor1.4",          # Antenna type
    max_transmit_gain_dBi=30.0,   # Maximum antenna gain in dBi
    orbits=[orbit_1, orbit_2]     # List of orbital parameters
)

# Criar gerador de números aleatórios
rng = np.random.RandomState(seed=42)

ngso_manager = StationFactory.generate_ngso_constellation(param, rng)

# Exibir informações da constelação
print("Tipo de estação:", ngso_manager.station_type)
print("Número de satélites:", ngso_manager.num_stations)
print("Primeiras coordenadas dos satélites:")
print("X:", ngso_manager.x[:5])
print("Y:", ngso_manager.y[:5])
print("Altura:", ngso_manager.height[:5])
