# -*- coding: utf-8 -*-
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase


@dataclass
class ParametersRas(ParametersBase):
    """
    Simulation parameters for Radio Astronomy Service
    """
    # x-y coordinates [m]
    x: float = 81000
    y: float = 0
    # antenna height [m]
    height: float = 15
    # Elevation angle [deg]
    elevation: float = 45
    # Azimuth angle [deg]
    azimuth: float = -90
    # center frequency [MHz]
    frequency: float = 43000
    # bandwidth [MHz]
    bandwidth: float = 1000
    # Antenna noise temperature [K]
    antenna_noise_temperature: float = 25
    # Receiver noise temperature [K]
    receiver_noise_temperature: float = 65
    # adjacent channel selectivity (dB)
    adjacent_ch_selectivity: float = 20
    # Antenna efficiency
    antenna_efficiency: float = 1
    # Antenna pattern of the FSS Earth station
    # Possible values: "ITU-R SA.509", "OMNI"
    antenna_pattern: str = "ITU-R SA.509"
    # Antenna gain for "OMNI" pattern
    antenna_gain: float = 0
    # Diameter of antenna [m]
    diameter: float = 15
    # Channel parameters
    # channel model, possible values are "FSPL" (free-space path loss),
    #                                    "TerrestrialSimple" (FSPL + clutter loss)
    #                                    "P452"
    channel_model: str = "P452"

    # P452 parameters
    # Total air pressure in hPa
    atmospheric_pressure: float = 935
    # Temperature in Kelvin
    air_temperature: float = 300
    # Sea-level surface refractivity (use the map)
    N0: float = 352.58
    # Average radio-refractive (use the map)
    delta_N: float = 43.127
    # Percentage p. Float (0 to 100) or RANDOM
    percentage_p: str = "0.2"
    # Distance over land from the transmit and receive antennas to the coast (km)
    Dct: float = 70
    # Distance over land from the transmit and receive antennas to the coast (km)
    Dcr: float = 70
    # Effective height of interfering antenna (m)
    Hte: float = 20
    # Effective height of interfered-with antenna (m)
    Hre: float = 3
    # Latitude of transmitter
    tx_lat: float = -23.55028
    # Latitude of receiver
    rx_lat: float = -23.17889
    # Antenna polarization
    polarization: str = "horizontal"
    # Determine whether clutter loss following ITU-R P.2108 is added (TRUE/FALSE)
    clutter_loss: bool = True

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
        if self.channel_model.upper() not in ["FSPL", "TERRESTRIALSIMPLE", "P452"]:
            raise ValueError(f"ParametersRas: \
                             Invalid value for parameter channel_model - {self.channel_model}. \
                             Allowed values are: \"FSPL\", \"TerrestrialSimple\", \"P452\"")
        if self.antenna_pattern.upper() not in ["ITU-R SA.509", "OMNI"]:
            raise ValueError(f"ParametersRas: \
                             Invalid value for parameter antenna_pattern - {self.antenna_pattern}. \
                             Allowed values are: \"ITU-R SA.509\", \"OMNI\"")
        if self.polarization.lower() not in ["horizonal", "vertical"]:
            raise ValueError(f"ParametersRas: \
                             Invalid value for parameter polarization - {self.polarization}. \
                             Allowed values are: \"horizontal\", \"vertical\"")