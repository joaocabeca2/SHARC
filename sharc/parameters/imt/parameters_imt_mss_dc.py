# Parameters for the IMT MSS-DC topology.
from dataclasses import dataclass, field
import numpy as np

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit

@dataclass
class ParametersImtMssDc(ParametersBase):
    """Dataclass for the IMT MSS-DC topology parameters."""
    section_name: str = "imt_mss_dc"

    nested_parameters_enabled = True

    # MSS_D2D system name
    name: str = "SystemA"

    # Orbit parameters
    orbits: list[ParametersOrbit] = field(default_factory=lambda: [ParametersOrbit()])

    # Number of beams
    num_beams: int = 19

    # Beam radius in meters
    # The beam radius should be calculated based on the Antenna Pattern used for IMT Space Stations
    beam_radius: float = 36516.0

    # Minimum elevation angle for satellite visibility
    min_elev_angle: float = 5.0

    def validate(self, ctx: str):
        """
        Raises
        ------
        ValueError
            If a parameter is not valid.
        """
        # Now do the sanity check for some parameters
        if self.num_beams not in [1, 7, 19]:
            raise ValueError(f"{ctx}.num_beams: Invalid number of sectors {self.num_sectors}")

        if self.beam_radius <= 0:
            raise ValueError(f"{ctx}.beam_radius: cell_radius must be greater than 0, but is {self.cell_radius}")
        else:
            self.cell_radius = self.beam_radius
            self.intersite_distance = np.sqrt(3) * self.cell_radius

