import pathlib
import yaml
from dotmap import DotMap

config_file = pathlib.Path(__file__).parent.resolve() / "parameters.yaml"

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

params = DotMap(config)

print(params)

# list of constellations

# Read paramters
# Setup orbit positions - run orbit predictor
# Intantiate NGSO Constellation
#   station factory
#   StationManager
# Instatiate Earth Station
# for each time step do
#    for each snap-shot do
#       update constellation postisions, azithuths and evevations
#       update earth station
#       calculate coupling loss
#       calculate aggregate interferece (or othre metrics)
#       store in data frame
# Dump results
# End