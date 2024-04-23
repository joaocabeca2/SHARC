"""Parameters definitions for Hotopsot systems
"""
import configparser
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase

@dataclass

class ParametersHotspot(ParametersBase):
    """Dataclass containing the Hotspot system parameters
    """

    num_hotspots_per_cell = 1
    max_dist_hotspot_ue   = 100
    min_dist_bs_hotspot   = 0


    def load_parameters_from_file(self, config_file: str):
        """Load the parameters from file an run a sanity check

        Parameters
        ----------
        file_name : str
            the path to the configuration file

        Raises
        ------
        ValueError
            if a parameter is not valid
        """
        super().load_parameters_from_file(config_file)