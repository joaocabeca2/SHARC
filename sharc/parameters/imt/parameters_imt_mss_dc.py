# Parameters for the IMT MSS-DC topology.
from dataclasses import dataclass, field

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit

@dataclass
class ParametersImtMssDc(ParametersBase):
    """Dataclass for the IMT MSS-DC topology parameters."""
    section_name: str = "imt_mss_dc"

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
