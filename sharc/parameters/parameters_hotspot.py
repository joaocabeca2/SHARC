# -*- coding: utf-8 -*-
"""
Parameters definitions for Hotspot systems
"""
from dataclasses import dataclass
from sharc.parameters.parameters_base import ParametersBase

@dataclass
class ParametersHotspot(ParametersBase):
    num_hotspots_per_cell: int = 1
    max_dist_hotspot_ue: int = 100
    min_dist_bs_hotspot: int = 0   

    def load_parameters_from_file(self, config_file: str):
        """
        Load the parameters from a file and run a sanity check.

        Parameters
        ----------
        config_file : str
            The path to the configuration file.

        Raises
        ------
        ValueError
            If a parameter is not valid.
        """
        super().load_parameters_from_file(config_file)
