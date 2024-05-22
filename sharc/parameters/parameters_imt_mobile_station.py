# -*- coding: utf-8 -*-
"""Parameters definitions for IMT Mobile Station systems
"""
from dataclasses import dataclass
from sharc.parameters.parameters_base import ParametersBase

@dataclass
class ParametersImtMobileStation(ParametersBase):
    """
    Simulation parameters for IMT Mobile Station network topology.
    """
    section_name: str = "IMT"
    
    # x-y coordinates [m]
    x: float = 500000
    y: float = 0
    
    # Antenna height [m]
    height: float = 0
    
    # Elevation angle [deg]
    elevation: float = 0
    
    # Azimuth angle [deg]
    azimuth: float = 180
    
    # Distribution settings
    distribution_enable: str = "OFF"
    distribution_type: str = "UNIFORM"
    azimuth_distribution: list = (-180, 180)
    elevation_distribution: list = (0, 50)
    
    # Center frequency [MHz]
    frequency: float = 2680
    
    # Bandwidth [MHz]
    bandwidth: float = 20
    
    # System receive noise temperature [K]
    noise_temperature: float = 627.06
    
    # Peak transmit power spectral density (clear sky) [dBW/Hz]
    tx_power_density: float = -70.79
    
    # Adjacent channel selectivity [dB]
    acs: float = 0
    
    # Antenna pattern of the fixed wireless service
    antenna_pattern: str = "OMNI"
    
    # Antenna peak gain [dBi]
    antenna_gain: float = 0
    
    # Channel model
    channel_model: str = "FSPL"
    
    # P.619 Parameters - NTN Topology
    altitude: float = 20000
    hibs_lat_deg: float = -15.8
    system_altitude: float = 8
    system_lat_deg: float = -15.45
    system_long_diff_deg: float = 0
    season: str = "SUMMER"
    
    
    
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

        # Sanity checks for some parameters
        if self.distribution_enable not in ["ON", "OFF"]:
            raise ValueError(f"ParametersImtMobileStation: Invalid value for distribution_enable {self.distribution_enable}")
        
        if self.distribution_type not in ["UNIFORM", "UNIFORM_NORMAL"]:
            raise ValueError(f"ParametersImtMobileStation: Invalid value for distribution_type {self.distribution_type}")

        if self.frequency <= 0:
            raise ValueError(f"ParametersImtMobileStation: frequency must be greater than 0, but is {self.frequency}")
        
        if self.bandwidth <= 0:
            raise ValueError(f"ParametersImtMobileStation: bandwidth must be greater than 0, but is {self.bandwidth}")
        
        if self.noise_temperature <= 0:
            raise ValueError(f"ParametersImtMobileStation: noise_temperature must be greater than 0, but is {self.noise_temperature}")
        
        if self.tx_power_density >= 0:
            raise ValueError(f"ParametersImtMobileStation: tx_power_density must be less than 0, but is {self.tx_power_density}")
        
        if not isinstance(self.acs, (int, float)):
            raise ValueError(f"ParametersImtMobileStation: acs must be a number, but is {self.acs}")
        
        if self.antenna_pattern not in ["OMNI", "BEAMFORMING"]:
            raise ValueError(f"ParametersImtMobileStation: Invalid value for antenna_pattern {self.antenna_pattern}")

        if not isinstance(self.antenna_gain, (int, float)):
            raise ValueError(f"ParametersImtMobileStation: antenna_gain must be a number, but is {self.antenna_gain}")
        
        if self.channel_model not in ["FSPL", "P619"]:
            raise ValueError(f"ParametersImtMobileStation: Invalid value for channel_model {self.channel_model}")

        if self.altitude <= 0:
            raise ValueError(f"ParametersImtMobileStation: altitude must be greater than 0, but is {self.altitude}")
        
        if self.hibs_lat_deg < -90 or self.hibs_lat_deg > 90:
            raise ValueError(f"ParametersImtMobileStation: hibs_lat_deg must be between -90 and 90, but is {self.hibs_lat_deg}")
        
        if self.system_altitude <= 0:
            raise ValueError(f"ParametersImtMobileStation: system_altitude must be greater than 0, but is {self.system_altitude}")
        
        if self.system_lat_deg < -90 or self.system_lat_deg > 90:
            raise ValueError(f"ParametersImtMobileStation: system_lat_deg must be between -90 and 90, but is {self.system_lat_deg}")

        if not isinstance(self.system_long_diff_deg, (int, float)):
            raise ValueError(f"ParametersImtMobileStation: system_long_diff_deg must be a number, but is {self.system_long_diff_deg}")
