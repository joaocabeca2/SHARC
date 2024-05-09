# -*- coding: utf-8 -*-
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase


@dataclass
class ParametersArns(ParametersBase):
    """
    Simulation parameters for Aeronautical Radionavigation Service
    """
    section_name: str = "ARNS"
    # x-y coordinates [m]
    x: float = 0.0
    y: float = 0.0
    ###########################################################################
    # Antenna height [m]
    height: float = 30.0
    ###########################################################################
    # Elevation angle [deg]
    elevation: float = 0.0
    ###########################################################################
    # Azimuth angle [deg]
    azimuth: float = 0.0
    ######## Creates a statistical distribution of azimuth and elevation########
    #### following variables azimuth_distribution and elevation_distribution####
    # if distribution_enable = ON, azimuth and elevation will vary statistically
    # distribution_type = UNIFORM , UNIFORM_NORMAL
    # UNIFORM = UNIFORM distribution in azimuth and elevation
    # 			- azimuth_distribution = initial azimuth, final azimuth
    # 			- elevation_distribution = initial elevation, final elevation
    # UNIFORM_NORMAL = UNIFORM and NORMAL distribution in azimuth and elevation,
    #                  respectivelly.
    # - azimuth_distribution = initial azimuth, final azimuth
    # 			- elevation_distribution = median, variance
    distribution_enable: bool = True
    distribution_type: str = "UNIFORM"
    azimuth_distribution: tuple = (-180, 180)
    elevation_distribution: tuple = (-2, 60)
    ###########################################################################
    # center frequency [MHz]
    frequency: float = 2700.5
    ###########################################################################
    # bandwidth [MHz]
    bandwidth: float = 0.5
    ###########################################################################
    # System receive noise temperature [K]
    noise_temperature: float = 2013.552
    ###########################################################################
    # Peak transmit power spectral density (clear sky) [dBW/Hz]
    tx_power_density: float = -70.79
    ###########################################################################
    # Adjacent channel selectivity [dB]
    acs: float = 30.0
    ###########################################################################
    # Antenna pattern of the fixed wireless service
    # Possible values: "OMNI", "UNIFORM, "COSINE, "COSSECANT SQUARED",
    # PHASED ARRAY
    antenna_pattern: str = "COSINE"
    ###########################################################################
    # antenna peak gain [dBi]
    antenna_gain: float = 38.0
    ###########################################################################
    ##################### Radar Antenna Pattern Parameters ####################
    ##################### UNIFORM, COSINE , COSECANT SQUARED####################
    beamwidth_el: float = 2.0
    beamwidth_az: float = 2.0
    ######## ATC Radar - Cossecant Squared Antenna Pattern parameters #########
    maximum_csc2_angle: float = 40.0
    highbeam_csc2: float = 0.0
    ############## Phased Array Antenna Pattern parameters ####################
    element_space: float = 0.5
    number_elements: int = 30
    ###########################################################################
    # Channel parameters
    # channel model, possible values are "FSPL" (free-space path loss),
    #                                    "TerrestrialSimple" (FSPL + clutter loss)
    channel_model: str = "FSPL"

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
        if self.distribution_type not in ["UNIFORM" , "UNIFORM_NORMAL"]:
            raise ValueError(f"ParametersArns: Invalid antenna azimuth distributioin type {self.distribution_type}")
        if self.antenna_pattern.upper() not in ["OMNI", "UNIFORM", "COSINE", "COSSECANT SQUARED", "PHASED ARRAY"]:
            raise ValueError(f"ParametersArns: Invalid Radar antenna pattern {self.antenna_pattern}")
        
        if self.channel_model.upper() not in ["FSPL", "SatelliteSimple", "P619"]:
            raise ValueError(f"ParametersArns: \
                             Invalid value for paramter channel_model = {self.channel_model}. \
                             Possible values are \"FSPL\", \"SatelliteSimple\", \"P619\".")
