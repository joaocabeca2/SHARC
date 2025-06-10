from sharc.parameters.parameters_base import ParametersBase

from dataclasses import dataclass


@dataclass
class ParametersAntennaWithFreq(ParametersBase):
    frequency: float = None

    # ideally we would make this happen at the ParametersBase.load_subparameters, but
    # since parser is a bit hacky, this is created without acces to system freq and antenna gain
    # so validation needs to happen manually afterwards
    def validate(self, ctx):
        if None in [
            self.frequency,
        ]:
            raise ValueError(f"{ctx} needs to have all its parameters set")

        if not isinstance(self.frequency, int) and not isinstance(self.frequency, float):
            raise ValueError(f"{ctx}.frequency needs to be a number")
