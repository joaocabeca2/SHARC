import typing
from dataclasses import field, dataclass
from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.wifi.parameters_hotspot import ParametersHotspot

@dataclass
class ParametersWifiTopology(ParametersBase):
    type: typing.Literal["HOTSPOT"
    ] = "HOTSPOT"

    hotspot: ParametersHotspot = field(default_factory=ParametersHotspot)

    def validate(self, ctx):
        if self.type == "HOTSPOT":
            self.hotspot.validate(f"{ctx}.hotspot")
        else:
            raise NotImplementedError(f"{ctx}.type == '{self.type}' may not be implemented")