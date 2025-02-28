import typing
from dataclasses import field, dataclass

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.imt.parameters_hotspot import ParametersHotspot
from sharc.parameters.imt.parameters_indoor import ParametersIndoor
from sharc.parameters.imt.parameters_macrocell import ParametersMacrocell
from sharc.parameters.imt.parameters_ntn import ParametersNTN
from sharc.parameters.imt.parameters_single_bs import ParametersSingleBS


@dataclass
class ParametersImtTopology(ParametersBase):
    is_spherical: bool = False  # whether we set the BS over a sphere (Earth) surface
    central_latitude: float = None
    central_longitude: float = None
    central_altitude: float = None

    type: typing.Literal[
        "MACROCELL", "HOTSPOT", "INDOOR", "SINGLE_BS", "NTN"
    ] = "MACROCELL"

    macrocell: ParametersMacrocell = field(default_factory=ParametersMacrocell)
    hotspot: ParametersHotspot = field(default_factory=ParametersHotspot)
    indoor: ParametersIndoor = field(default_factory=ParametersIndoor)
    single_bs: ParametersSingleBS = field(default_factory=ParametersSingleBS)
    ntn: ParametersNTN = field(default_factory=ParametersNTN)

    def validate(self, ctx):
        if self.is_spherical:
            if None in [self.central_altitude, self.central_latitude, self.central_longitude]:
                raise ValueError(
                    "When imt topology should be considered spherical, you need to pass" +
                    "central_altitude, central_latitude and central_longitude" +
                    f"on {ctx}"
                )

        match self.type:
            case "MACROCELL":
                self.macrocell.validate(f"{ctx}.macrocell")
            case "HOTSPOT":
                self.hotspot.validate(f"{ctx}.hotspot")
            case "INDOOR":
                self.indoor.validate(f"{ctx}.indoor")
            case "SINGLE_BS":
                self.single_bs.validate(f"{ctx}.single_bs")
            case "NTN":
                self.ntn.validate(f"{ctx}.ntn")
            case _:
                raise NotImplementedError(f"{ctx}.type == '{self.type}' may not be implemented")
