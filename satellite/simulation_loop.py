import pathlib
import logging
import yaml
from dotmap import DotMap
from types import List
import numpy as np
import pandas as pd

from satellite.ngso.ngso_constellation import NgsoConstellation

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# data frame columns
results_df_cols = [
    "index",
    "time_step",
    "constellation_name",
    "orbit_idx",
    "plane_idx",
    "satellite_idx",
    "sx",
    "sy",
    "sz",
    "azimuth",
    "elevation",
    "path_loss",
    "interference",
    "rx_gain",
    "tx_gain",
    "pfd"
]


def run():
    """Implements the main simulation loop"""

    logger.info("STARTING SIMULATION")

    # Read simulation paramters
    config_file = pathlib.Path(__file__).parent.resolve() / "parameters.yaml"
    logger.info(f"Reading simulation parameters from {config_file}")

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # TODO: Add parameter sanitization!
    params = DotMap(config)

    logger.debug(params.pprint())

    # Load the NGSO constellations
    constellations = List[NgsoConstellation]
    for _, constellation_param in enumerate(params.constellations):
        logger.debug(f"Loading constellation {constellation_param.name}")
        constellations.append(NgsoConstellation(constellation_param))

    for const in constellations:
        const.initialize()
        const.calculate_orbits(params.time_step_sec, params.num_orbital_periods)

    # Earth Station Initialization goes here

    # Simulation main loop
    n_time_steps = np.floor(params.simulation.simulation_time_span_sec / params.simulation.simulation_time_step_sec)
    for t in range(n_time_steps):
        # Update all space-station positions
        for c in constellations:
            c.update_orbits()
        for snap_shot in range(params.simulation.num_snapshots):
            # update earth-station geometry
            # calculate coupling loss
            # calculate aggregate interference (or other metrics)
            # update results
            pass
