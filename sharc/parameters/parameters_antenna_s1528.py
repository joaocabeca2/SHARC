import configparser
from dataclasses import dataclass
from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_p619 import ParametersP619


@dataclass
class ParametersAntennaS1528(ParametersBase):
    """Class to handle parameters for AntennaS1528Taylor, AntennaS1528Leo, and AntennaS1528"""
    antenna_gain: float = 0.0
    frequency: float = 0.0
    bandwidth: float = 0.0
    slr: float = 0.0
    n_side_lobes: int = 0
    l_r: float = 0.0
    l_t: float = 0.0
    roll_off: int = 0
    antenna_3_dB: float = 0.0
    antenna_l_s: float = 0.0
    antenna_pattern: str = "FSS_SS"

    param_p619 = ParametersP619()
    space_station_alt_m: float = 35780000.0
    earth_station_alt_m: float = 0.0
    earth_station_lat_deg: float = 0.0
    earth_station_long_diff_deg: float = 0.0
    season: str = "SUMMER"
    # Channel parameters
    # channel model, possible values are "FSPL" (free-space path loss),
    #                                    "SatelliteSimple" (FSPL + 4 + clutter loss)
    #                                    "P619"
    channel_model: str = "P619"
    # The required near-in-side-lobe level (dB) relative to peak gain
    # according to ITU-R S.672-4
    antenna_l_s: float = -20.0
    # 3 dB beamwidth angle (3 dB below maximum gain) [degrees]
    antenna_3_dB: float = 0.65

    def load_parameters_from_file(self, config_file: str):
        """Load the parameters from file and perform sanity checks specific to antenna parameters"""
        super().load_parameters_from_file(config_file)
        # Add any additional sanity checks specific to antenna parameters here

        # Now do the sanity check for some parameters
        if self.antenna_pattern not in ["ITU-R S.672", "ITU-R S.1528", "FSS_SS", "OMNI"]:
            raise ValueError(f"ParametersFssSs: \
                             invalid value for parameter antenna_pattern - {self.antenna_pattern}. \
                             Possible values \
                             are \"ITU-R S.672\", \"ITU-R S.1528\", \"FSS_SS\", \"OMNI\"")

        if self.season.upper() not in ["SUMMER", "WINTER"]:
            raise ValueError(f"ParametersFssSs: \
                             Invalid value for parameter season - {self.season}. \
                             Possible values are \"SUMMER\", \"WINTER\".")

        if self.channel_model.upper() not in ["FSPL", "SATELLITESIMPLE", "P619"]:
            raise ValueError(f"ParametersFssSs: \
                             Invalid value for paramter channel_model = {self.channel_model}. \
                             Possible values are \"FSPL\", \"SatelliteSimple\", \"P619\".")
        self.param_p619.load_from_paramters(self)
