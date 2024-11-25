# -*- coding: utf-8 -*-
from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase


@dataclass
class ParametersP1411(ParametersBase):
    """Dataclass containing the P.1411-2 propagation model parameters
    """
    environment: str = 'URBAN'
    frequency: float = 7
    #Corner loss in dB
    l_corner: float = 20

    d_corner: float = 30
    #street width at the position of the Station 1 (m)
    street_width1: float = 0
    #street width at the position of the Station 2 (m)
    street_width2: float = 0
    #distance Station 1 to street crossing (m)
    distance1: float = 0
    #distance Station 2 to street crossing (m)
    distance2: float = 0
    #is the corner angle (rad)
    corner_angle: float = 0
    wavelength = 3e8 / (frequency*1e9)

    def load_from_paramters(self, param: ParametersBase):
        """Used to load parameters of P.1411-2 from IMT or system parameters

        Parameters
        ----------
        param : ParametersBase
            IMT or system parameters
        """
        self.environment = param.environment
        self.l_corner = param.l_corner
        self.d_corner = param.d_corner
        self.street_width1 = param.street_width1
        self.street_width2 = param.street_width2
        self.distance1 = param.distance1
        self.distance2 = param.distance2
        self.corner_angle = param.corner_angle
        self.wavelength = param.wavelength