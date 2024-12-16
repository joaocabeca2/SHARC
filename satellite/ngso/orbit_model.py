"""Implements a Space Station Orbit model as described in Rec. ITU-R S.1325-3
"""

import numpy as np

from custom_functions import wrap2pi, eccentric_anomaly, keplerian2eci, eci2ecef, plot_ground_tracks
from constants import EARTH_RADIUS_KM, KEPLER_CONST, EARTH_ROTATION_RATE

class OrbitModel():
    def __init__(self,
                 Nsp: int,
                 Np: int,
                 phasing: float,
                 long_asc: float,
                 omega: float,
                 delta: float,
                 hp: float,
                 ha: float,
                 Mo: float):
        """Instantiates and OrbitModel object from the Orbit parameters as specified in S.1529

        Parameters
        ----------
        Nsp : int
            number of satellites in the orbital plane (A.4.b.4.b)
        Np : int
            number of orbital planes (A.4.b.2)
        phasing : float
            satellite phasing between planes, in degrees
        long_asc : float
            initial longitude of ascending node of the first plane, in degrees
        omega : float
            argument of perigee, in degrees
        delta : float
            orbital plane inclination, in degrees
        hp : float
            altitude of perigee in km
        ha : float
            altitude of apogee in km
        Mo : float
            initial mean anomaly for first satellite of first plane, in degrees
        """
        self.Nsp = Nsp
        self.Np = Np
        self.phasing = phasing
        self.long_asc = long_asc
        self.omega = omega
        self.delta = delta
        self.hp = hp
        self.ha = ha
        self.Mo = Mo

        # Derive other orbit parameters
        self.semi_major_axis = (hp + ha + 2 * EARTH_RADIUS_KM) / 2  # semi-major axis, in km
        self.eccentricity = (ha + EARTH_RADIUS_KM - self.semi_major_axis) / self.semi_major_axis  # orbital eccentricity (e)
        self.orbital_period_sec = 2 * np.pi * np.sqrt(self.semi_major_axis ** 3 / KEPLER_CONST)  # orbital period, in seconds
        self.sat_sep_angle_deg = 360 / self.Nsp  # satellite separation angle in the plane (degrees)
        self.orbital_plane_inclination = 360 / self.Np  # angle between plane intersections with the equatorial plane (degrees)

        # Initial mean anomalies for all the satellites
        # shape (Np, Nsp)
        self.initial_mean_anomalies = (Mo + np.arange(Nsp) * self.sat_sep_angle_deg + np.arange(self.Np)[:, None] *
                                       self.phasing) % 360
        self._initial_mean_anomalies_flat = np.radians(self.initial_mean_anomalies.flatten())  # shape (Np*Nsp,)

        # Initial longitudes of ascending node for all the planes
        # shape (Np, Nsp)
        self.Omega_o = (self.long_asc + np.arange(self.Nsp) * 0 + np.arange(self.Np)[:, None] *
                        self.orbital_plane_inclination) % 360
        self.Omega0 = np.radians(self.Omega_o.flatten())  # shape (Np*Nsp,)

    def get_satellite_positions_time_interval(self, initial_time_secs=0, interval_secs=5, n_periods=4) -> dict:
        """Return the orbit positions vector
        """
        t = np.arange(initial_time_secs, n_periods * self.orbital_period_sec + interval_secs, interval_secs)
        return self.__get_satellite_positions(t)

    def get_orbit_positions_time_instant(self, time_instant_secs=0) -> dict:
        """Returns satellite positions in a determined time instant in seconds"""
        t = np.array([time_instant_secs])
        return self.__get_satellite_positions(t)

    def get_orbit_positions_random_time(self, rng: np.random.RandomState) -> dict:
        return self.__get_satellite_positions(rng.random_sample(1) * self.orbital_period_sec)

    def __get_satellite_positions(self, t: np.array) -> dict:
        # Mean anomaly (M)
        mean_anomaly = (self._initial_mean_anomalies_flat[:, None] +
                        (2 * np.pi / self.orbital_period_sec) * t) % (2 * np.pi)

        # Eccentric anomaly (E)
        eccentric_anom = eccentric_anomaly(self.eccentricity, mean_anomaly)

        # True anomaly (v)
        v = 2 * np.arctan(np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * np.tan(eccentric_anom / 2))
        v = np.mod(v, 2 * np.pi)

        # Distance of the satellite to Earth's center (r)
        r = self.semi_major_axis * (1 - self.eccentricity ** 2) / (1 + self.eccentricity * np.cos(v))

        # True anomaly relative to the line of nodes (gamma)
        # gamma = wrap2pi(v + np.radians(self.omega))  # gamma in the interval [-pi, pi]

        # Latitudes of the satellites, in radians (theta)
        # theta = np.arcsin(np.sin(gamma) * np.sin(np.radians(self.delta)))

        # Longitude variation due to angular displacement, in radians (phiS)
        # phiS = np.arccos(np.cos(gamma) / np.cos(theta)) * np.sign(gamma)

        # Longitudes of the ascending node (OmegaG)
        OmegaG = (self.Omega0[:, None] + EARTH_ROTATION_RATE * t)  # shape (Np*Nsp, len(t))
        OmegaG = wrap2pi(OmegaG)

        # POSITION CALCULATION IN ECEF COORDINATES - ITU-R S.1503
        r_eci = keplerian2eci(self.semi_major_axis, self.eccentricity, self.orbital_plane_inclination,
                              np.degrees(self.Omega0), self.omega, np.degrees(v))

        r_ecef = eci2ecef(t, r_eci)
        sx, sy, sz = r_ecef[0], r_ecef[1], r_ecef[2]
        lat = np.degrees(np.arcsin(sz / r))
        lon = np.degrees(np.arctan2(sy, sx))

        pos_vector = {
            'lat': lat,
            'lon': lon,
            'sx': sx,
            'sy': sy,
            'sz': sz
        }
        return pos_vector


if __name__ == "__main__":
    # Plot Global Star orbit using OrbitModel object
    orbit = OrbitModel(
        Nsp=6,
        Np=8,
        phasing=7.5,
        long_asc=0,
        omega=0,
        delta=52,
        hp=1414,
        ha=1414,
        Mo=0
    )

    # pos_vec = orbit.get_orbit_positions_vec()
    # pos_vec = orbit.get_orbit_positions_time_instant(time_instant_secs=10)
    pos_vec = orbit.get_orbit_positions_random_time(rng=np.random.RandomState(seed=10))
    plot_ground_tracks(pos_vec['lat'], pos_vec['lon'], planes=[1, 2, 3, 4, 5, 6, 7, 8], satellites=[1])
