from dataclasses import dataclass

from sharc.parameters.parameters_base import ParametersBase

class ParametersNGSO(ParametersBase):

    Nsp: int = 6  # number of satellites in the orbital plane (A.4.b.4.b)
    Np: int = 8  # number of orbital planes (A.4.b.2)
    phasing: float = 7.5  # satellite phasing between planes, in degrees
    long_asc: int = 0  # initial longitude of ascending node of the first plane, in degrees (A.4.b.4.j)
    omega: int = 0  # argument of perigee, in degrees (A.4.b.4.i)
    delta: int = 52  # orbital plane inclination, in degrees (A.4.b.4.a)
    hp: int = 1414  # altitude of perigee in km (A.4.b.4.e)
    ha: int = 1414  # altitude of apogee in km (A.4.b.4.d)
    Mo: int = 0  # initial mean anomaly for first satellite of first plane, in degrees
