# -*- coding: utf-8 -*-
import configparser
from dataclasses import dataclass

from sharc.sharc_definitions import SHARC_IMPLEMENTED_SYSTEMS

@dataclass
class ParametersGeneral:
    """Dataclass containing the general parameters for the simulator
    """
    section_name: str = "GENERAL"
    num_snapshots: int = 10000
    imt_link: str = "DOWNLINK"
    system: str = "RAS"
    enable_cochannel: bool = False
    enable_adjacent_channel: bool = True
    seed: int = 101
    overwrite_output: bool = True

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
        config = configparser.ConfigParser()
        config.read(config_file)

        self.num_snapshots = config.getint(self.section_name, "num_snapshots")
        self.imt_link = config.get(self.section_name, "imt_link")
        if self.imt_link not in ["DOWNLINK", "UPLINK"]:
            raise ValueError(f"ParametersGeneral: \
                             invalid value for parameter imt_link - {self.imt_link} \
                             Possible values are DOWNLINK and UPLINK")
        self.system = config.get(self.section_name, "system")
        if self.system not in SHARC_IMPLEMENTED_SYSTEMS:
            raise ValueError(f"Invalid system name {self.system}")
        self.enable_adjacent_channel = config.getboolean(self.section_name,
                                                         "enable_adjacent_channel")
        self.enable_cochannel = config.getboolean(self.section_name,
                                                         "enable_cochannel")
        self.seed = config.getint(self.section_name, "seed")
        self.overwrite_output = config.getboolean(self.section_name, "overwrite_output")
