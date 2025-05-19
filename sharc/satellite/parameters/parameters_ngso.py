from dataclasses import dataclass, field
from typing import List
from sharc.parameters.parameters_base import ParametersBase
from parameters.parameters_orbit import ParametersOrbit


@dataclass
class ParametersNgsoConstellation(ParametersBase):
    section_name: str = "ngso"
    name: str = "Default"
    orbits: List[ParametersOrbit] = field(default_factory=list)
    antenna: str = None
    max_gain_dbi: float = 0.0

    def load_parameters_from_file(self, config_file: str):
        """Load parameters from file and validate."""
        super().load_parameters_from_file(config_file)

        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"ParametersNgsoConstellation: Invalid name = {self.name}. \
                             Must be a non-empty string.")
        if self.antenna not in ["Taylor1.4", "ITU-R M.1466", "OMNI"]:
            raise ValueError(f"ParametersNgsoConstellation: Invalid antenna = {self.antenna}. \
                             Allowed values are \"Taylor1.4\", \"ITU-R M.1466\", \"OMNI\".")
        if self.max_gain_dbi < 0:
            raise ValueError(f"ParametersNgsoConstellation: Invalid max_gain_dbi = {self.max_gain_dbi}. \
                             Gain must be non-negative.")
        if not self.orbits:
            raise ValueError("ParametersNgsoConstellation: No orbits defined. \
                             At least one orbit must be specified.")
        for orbit in self.orbits:
            if not isinstance(orbit, ParametersOrbit):
                raise ValueError("ParametersNgsoConstellation: Invalid orbit configuration. \
                                 All orbits must be instances of ParametersOrbit.")
