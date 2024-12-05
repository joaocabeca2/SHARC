from dataclasses import dataclass, field
from typing import List

from sharc.parameters.parameters_base import ParametersBase

@dataclass
class ParametersOrbit():
    n_planes: int = 8  # number of orbital planes (A.4.b.2)
    sats_per_plane: int = 6  # number of satellites in the orbital plane (A.4.b.4.b)
    phasing_deg: float = 7.5  # satellite phasing between planes, in degrees
    long_asc_deg: float = 0  # initial longitude of ascending node of the first plane, in degrees (A.4.b.4.j)
    omega_deg: float = 0  # argument of perigee, in degrees (A.4.b.4.i)
    inclination_deg: float = 52.0  # orbital plane inclination, in degrees (A.4.b.4.a)
    pergee_alt_km: float = 1414.0  # altitude of perigee in km (A.4.b.4.e)
    apogee_alt_km: float = 1414.0  # altitude of apogee in km (A.4.b.4.d)
    initial_mean_anomaly: float = 0.0  # initial mean anomaly for first satellite of first plane, in degrees

@dataclass
class ParametersNgsoConstellation(ParametersBase):
    """
    Defines parameters for NGSO Constellation based on ITU S.1503 definitions.
    """
    section_name: str = "ngso"
    name: str = "Default"
    orbits: List[ParametersOrbit] = field(default_factory=ParametersOrbit)

    def add_orbit(self, param_orbit: ParametersOrbit):
        self.orbits.append(param_orbit)

    def get_num_of_shells(self) -> int:
        return len(self.orbits)

@dataclass
class ParametersConstellations(ParametersBase):
    section_name: str = "constellations"
    constellations: List[ParametersNgsoConstellation] = field(default_factory=ParametersNgsoConstellation)

if __name__ == "__main__":
    # Adding multiple shells to this constellation
    shell_1 = ParametersOrbit(n_planes=20,
                              inclination_deg=54.5,
                              pergee_alt_km=525,
                              apogee_alt_km=525,
                              sats_per_plane=32,
                              long_asc_deg=18,
                              phasing_deg=3.9)
    shell_2 = ParametersOrbit(n_planes=12,
                              inclination_deg=26,
                              pergee_alt_km=580,
                              apogee_alt_km=580,
                              sats_per_plane=20,
                              long_asc_deg=30,
                              phasing_deg=2.8)
    shell_3 = ParametersOrbit(n_planes=26,
                              inclination_deg=97.77,
                              pergee_alt_km=595,
                              apogee_alt_km=595,
                              sats_per_plane=30,
                              long_asc_deg=14,
                              phasing_deg=7.8)
    param_ngso = ParamtersNgsoConstellation("ACME-Constellation", [shell_1, shell_2, shell_3])
    print(param_ngso)
    print(param_ngso.get_num_of_shells())
