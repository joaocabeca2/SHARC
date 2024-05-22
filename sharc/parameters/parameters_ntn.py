# -*- coding: utf-8 -*-
"""Parameters definitions for NTN systems
"""
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase
import numpy as np

@dataclass
class ParametersNTN(ParametersBase):
    """
    Simulation parameters for NTN network topology.
    """
    section_name: str = "NTN"
    
    # Number of sectors in NTN network topology. Allowed: [1, 3, 7 or 19 sectors]
    num_sectors: int = 7
    
    # NTN clusters in network topology [m]. Allowed: 0, 1
    num_clusters: int = 0
    
    # NTN Airborne Platform height (m)
    bs_height: int = 20000
    
    # NTN cell radius in network topology [m]
    cell_radius: int = 90000
    
    # NTN Intersite Distance (m). Intersite distance = Cell Radius * sqrt(3)
    intersite_distance: int = 155884
    
    # Antenna azimuth per sector - NTN sector topology [degree]
    azimuth: np.array = np.arra([90, 210, 330])
    #azimuth7: tuple = (0, 0, 60, 120, 180, 240, 300)
    # azimuth19: tuple = (0, 15, 30, 45, 75, 90, 105, 135, 150, 165, 195, 210, 225, 255, 270, 285, 315, 330, 345)
    
    # Elevation (Mechanical Antenna Downtilt) per sector - NTN topology [degree]
    elevation: np.array = np.array([-90, -90, -90])
    #elevation7: tuple = (-90, -23, -23, -23, -23, -23, -23)
    # elevation19: tuple = (-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30)
    
    # Conducted power per element [dBm/bandwidth]
    bs_conducted_power: int = 37
    
    # Backoff Power [Layer 2] [dB]. Allowed: 7 sector topology - Layer 2
    bs_backoff_power: int = 3
    
    # NTN Antenna configuration
    bs_n_rows_layer1: int = 2
    bs_n_columns_layer1: int = 2
    bs_n_rows_layer2: int = 4
    bs_n_columns_layer2: int = 2
    
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

        # Now do the sanity check for some parameters
        if self.num_sectors not in [1, 3, 7, 19]:
            raise ValueError(f"ParametersNTN: Invalid number of sectors {self.num_sectors}")

        if self.num_clusters not in [0, 1]:
            raise ValueError(f"ParametersNTN: Invalid number of clusters {self.num_clusters}")

        if self.bs_height <= 0:
            raise ValueError(f"ParametersNTN: bs_height must be greater than 0, but is {self.bs_height}")
        
        if self.cell_radius <= 0:
            raise ValueError(f"ParametersNTN: cell_radius must be greater than 0, but is {self.cell_radius}")
        
        if self.intersite_distance <= 0:
            raise ValueError(f"ParametersNTN: intersite_distance must be greater than 0, but is {self.intersite_distance}")
        
        if not isinstance(self.bs_conducted_power, int) or self.bs_conducted_power <= 0:
            raise ValueError(f"ParametersNTN: bs_conducted_power must be a positive integer, but is {self.bs_conducted_power}")
        
        if not isinstance(self.bs_backoff_power, int) or self.bs_backoff_power < 0:
            raise ValueError(f"ParametersNTN: bs_backoff_power must be a non-negative integer, but is {self.bs_backoff_power}")
        
        if self.num_sectors == 7:
            if self.bs_n_rows_layer1 != 2 or self.bs_n_columns_layer1 != 2:
                raise ValueError(f"ParametersNTN: For 7 sector topology, Layer 1 must have 2 rows and 2 columns")
            if self.bs_n_rows_layer2 != 4 or self.bs_n_columns_layer2 != 2:
                raise ValueError(f"ParametersNTN: For 7 sector topology, Layer 2 must have 4 rows and 2 columns")
