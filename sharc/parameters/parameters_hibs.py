# -*- coding: utf-8 -*-
"""Parameters definitions for HIBS systems
"""
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase

@dataclass
class ParametersHIBS(ParametersBase):
    """
    Simulation parameters for HIBS network topology.
    """
    section_name: str = "HIBS"
    
    # Number of sectors in HIBS network topology. Allowed: [1, 3, 7 or 19 sectors]
    num_sectors: int = 7
    
    # HIBs clusters in network topology [m]. Allowed: 0, 1
    num_clusters: int = 0
    
    # HIBS Airborne Platform height (m)
    bs_height: int = 20000
    
    # HIBS cell radius in network topology [m]
    cell_radius: int = 90000
    
    # HIBS Intersite Distance (m). Intersite distance = Cell Radius * sqrt(3)
    intersite_distance: int = 155884
    
    # Antenna azimuth per sector - HIBS sector topology [degree]
    azimuth3: tuple = (90, 210, 330)
    azimuth7: tuple = (0, 0, 60, 120, 180, 240, 300)
    #azimuth19: tuple = (0, 15, 30, 45, 75, 90, 105, 135, 150, 165, 195, 210, 225, 255, 270, 285, 315, 330, 345)
    
    # Elevation (Mechanical Antenna Downtilt) per sector - HIBS topology [degree]
    elevation3: tuple = (-90, -90, -90)
    elevation7: tuple = (-90, -23, -23, -23, -23, -23, -23)
    #elevation19: tuple = (-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30)
    
    # Conducted power per element [dBm/bandwidth]
    bs_conducted_power: int = 37
    
    # Backoff Power [Layer 2] [dB]. Allowed: 7 sector topology - Layer 2
    bs_backoff_power: int = 3
    
    # HIBs Antenna configuration
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
            raise ValueError(f"ParametersHIBS: Invalid number of sectors {self.num_sectors}")

        if self.num_clusters not in [0, 1]:
            raise ValueError(f"ParametersHIBS: Invalid number of clusters {self.num_clusters}")

        if self.bs_height <= 0:
            raise ValueError(f"ParametersHIBS: bs_height must be greater than 0, but is {self.bs_height}")
        
        if self.cell_radius <= 0:
            raise ValueError(f"ParametersHIBS: cell_radius must be greater than 0, but is {self.cell_radius}")
        
        if self.intersite_distance <= 0:
            raise ValueError(f"ParametersHIBS: intersite_distance must be greater than 0, but is {self.intersite_distance}")
        
        if not isinstance(self.bs_conducted_power, int) or self.bs_conducted_power <= 0:
            raise ValueError(f"ParametersHIBS: bs_conducted_power must be a positive integer, but is {self.bs_conducted_power}")
        
        if not isinstance(self.bs_backoff_power, int) or self.bs_backoff_power < 0:
            raise ValueError(f"ParametersHIBS: bs_backoff_power must be a non-negative integer, but is {self.bs_backoff_power}")
        
        if self.num_sectors == 7:
            if self.bs_n_rows_layer1 != 2 or self.bs_n_columns_layer1 != 2:
                raise ValueError(f"ParametersHIBS: For 7 sector topology, Layer 1 must have 2 rows and 2 columns")
            if self.bs_n_rows_layer2 != 4 or self.bs_n_columns_layer2 != 2:
                raise ValueError(f"ParametersHIBS: For 7 sector topology, Layer 2 must have 4 rows and 2 columns")
