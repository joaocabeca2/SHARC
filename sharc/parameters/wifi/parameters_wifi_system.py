# -*- coding: utf-8 -*-
"""Parameters definitions for WiFi systems
"""
from dataclasses import dataclass, field

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_p452 import ParametersP452
from sharc.parameters.wifi.parameters_wifi_topology import ParametersWifiTopology

@dataclass
class ParametersWifiSystem(ParametersBase):
    """Dataclass containing the WiFi system parameters
    """
    section_name: str = "wifi"

    # type of FSS-ES location:
    # FIXED - position must be given
    # CELL - random within central cell
    # NETWORK - random within whole network
    # UNIFORM_DIST - uniform distance from cluster centre,
    #                between min_dist_to_bs and max_dist_to_bs
    location: str = "UNIFORM_DIST"
    # x-y coordinates [m] (only if FIXED location is chosen)
    x: float = 10000.0
    y: float = 0.0
    # minimum distance from BSs [m]
    min_dist_to_bs: float = 10.0
    # maximum distance from centre BSs [m] (only if UNIFORM_DIST is chosen)
    max_dist_to_bs: float = 10.0
    # antenna height [m]
    height: float = 6.0
    # Elevation angle [deg], minimum and maximum, values
    elevation_min: float = 48.0
    elevation_max: float = 80.0
    # Azimuth angle [deg]
    # either a specific angle or string 'RANDOM'
    azimuth: str = 0.2
    # center frequency [MHz]
    frequency: float = 43000.0
    # bandwidth [MHz]
    bandwidth: float = 6.0
    # adjacent channel selectivity (dB)
    adjacent_ch_selectivity: float = 0.0
    # Peak transmit power spectral density (clear sky) [dBW/Hz]
    tx_power_density: float = -68.3
    # System receive noise temperature [K]
    noise_temperature: float = 950.0
    # antenna peak gain [dBi]
    antenna_gain: float = 0.0
    # Antenna pattern of the FSS Earth station
    # Possible values: "ITU-R S.1855", "ITU-R S.465", "ITU-R S.580", "OMNI",
    #                  "Modified ITU-R S.465"
    antenna_pattern: str = "Modified ITU-R S.465"
    # Antenna envelope gain (dBi) - only relevant for "Modified ITU-R S.465" model
    antenna_envelope_gain: float = 0.0
    # Diameter of the antenna [m]
    diameter: float = 1.8
    # Channel parameters
    # channel model, possible values are "FSPL" (free-space path loss),
    #                                    "TerrestrialSimple" (FSPL + clutter loss)
    #                                    "P452"
    #                                    "TVRO-URBAN"
    #                                    "TVRO-SUBURBAN"
    #                                    "HDFSS"
    channel_model: str = "P452"
     # P452 parameters
    param_p452 = ParametersP452()
    # Total air pressure in hPa
    atmospheric_pressure: float = 935.0
    # Temperature in Kelvin
    air_temperature: float = 300.0
    # Sea-level surface refractivity (use the map)
    N0: float = 352.58
    # Average radio-refractive (use the map)
    delta_N: float = 43.127
    # Percentage p. Float (0 to 100) or RANDOM
    percentage_p: str = "0.2"
    # Distance over land from the transmit and receive antennas to the coast (km)
    Dct: float = 70.0
    # Distance over land from the transmit and receive antennas to the coast (km)
    Dcr: float = 70.0
    # Effective height of interfering antenna (m)
    Hte: float = 20.0
    # Effective height of interfered-with antenna (m)
    Hre: float = 3.0
    # Latitude of transmitter
    tx_lat: float = -23.55028
    # Latitude of receiver
    rx_lat: float = -23.17889
    # Antenna polarization
    polarization: str = "horizontal"
    # Determine whether clutter loss following ITU-R P.2108 is added (TRUE/FALSE)
    clutter_loss: bool = True

    topology: ParametersWifiTopology = field(default_factory=ParametersWifiTopology)

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

        if self.channel_model.upper() not in [
            "FSPL", "TERRESTRIALSIMPLE", "P452", "P619",
            "TVRO-URBAN", "TVRO-SUBURBAN", "HDFSS", "UMA", "UMI",
        ]:
            raise ValueError(
                f"ParametersFssEs: Invalid value for parameter channel_model - {self.channel_model}",
            )

        if self.channel_model == "P452":
            self.param_p452.load_from_paramters(self)

        elif self.channel_model == "P619":
            self.param_p619.load_from_paramters(self)
        

    

    