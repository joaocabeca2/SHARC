from sharc.parameters.parameters_mss_d2d import ParametersOrbit, ParametersMssD2d
from sharc.topology.topology_single_base_station_spherical import TopologySingleBaseStationSpherical
#from sharc.station_manager import StationManager
#from sharc.support.enumerations import StationType
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
param = ParametersMssD2d(
    name="Acme-Star-1",                         # Name of the constellation
    antenna_pattern="TITU-R-S.1528-Taylor",     # Antenna type
    antenna_gain=30.0,                 # Maximum antenna gain in dBi
    orbits=[orbit_1, orbit_2]                   # List of orbital parameters
)

# Creating an IMT topology
imt_topology = TopologySingleBaseStationSpherical(cell_radius=500,num_clusters=2,central_latitude=-15.7801,central_longitude=-47.9292)


# Criar gerador de números aleatórios
rng = np.random.RandomState(seed=42)


ngso_manager = StationFactory.generate_mss_d2d_multiple_orbits(param, rng,imt_topology)

# Exibir informações da constelação
print("Tipo de estação:", ngso_manager.station_type)
print("Número de satélites:", ngso_manager.num_stations)
print("Primeiras coordenadas dos satélites:")
print("X:", ngso_manager.x[:5])
print("Y:", ngso_manager.y[:5])
print("Altura:", ngso_manager.height[:5])
