import numpy as np
from sharc.support.enumerations import StationType
from parametetsNGSO import ParametersNGSO
from custom_functions import wrap2pi, eccentric_anomaly, keplerian2eci, eci2ecef
from constants import EARTH_RADIUS_KM, KEPLER_CONST, EARTH_ROTATION_RATE
from station_manager import StationManager
import pandas as pd

class StationFactory(object):


    @staticmethod
    def generate_satellite_station(param:ParametersNGSO,interval: int= 5, nP = 4 ):

        # time interval, in seconds
        # number of orbital periods of simulation

    


        # Orbital parameters
        a = (ParametersNGSO.hp + ParametersNGSO.ha + 2 * EARTH_RADIUS_KM) / 2  # semi-major axis, in km
        e = (ParametersNGSO.ha + EARTH_RADIUS_KM - a) / a  # orbital eccentricity (e)
        P = 2 * np.pi * np.sqrt(a ** 3 / KEPLER_CONST)  # orbital period, in seconds
        beta = 360 / ParametersNGSO.Nsp  # satellite separation angle in the plane (degrees)
        psi = 360 / ParametersNGSO.Np  # angle between plane intersections with the equatorial plane (degrees)
        t = np.arange(0, nP * P + interval, interval)  # vector of time, in seconds

        # Initial mean anomalies for all the satellites
        M_o = (ParametersNGSO.Mo + np.arange(ParametersNGSO.Nsp) * beta + np.arange(ParametersNGSO.Np)[:, None] * ParametersNGSO.phasing) % 360  # shape (Np, Nsp)
        M0 = np.radians(M_o.flatten())  # shape (Np*Nsp,)

        # Initial longitudes of ascending node for all the planes
        Omega_o = (ParametersNGSO.long_asc + np.arange(ParametersNGSO.Nsp) * 0 + np.arange(ParametersNGSO.Np)[:, None] * psi) % 360  # shape (Np, Nsp)
        Omega0 = np.radians(Omega_o.flatten())  # shape (Np*Nsp,)

        # INPUTS - EARTH SYSTEM

        # Station position vector, in km
        lat_es = [0, -10]  # latitudes of stations, in degrees
        lon_es = [0, -30]  # longitudes of stations, in degrees
        alt_es = [0, 0]  # altitudes of stations, in meters
        rx = EARTH_RADIUS_KM + alt_es[0] * 10**-3
        px = rx * np.cos(np.radians(lat_es[0])) * np.cos(np.radians(lon_es[0]))
        py = rx * np.cos(np.radians(lat_es[0])) * np.sin(np.radians(lon_es[0]))
        pz = rx * np.sin(np.radians(lat_es[0]))
        p = np.array([px, py, pz])

                
        # Display key results
        #print("Orbital parameters derived from ParametersNGSO:")
        #print(f"Semi-major axis (a): {a} km")
        #print(f"Eccentricity (e): {e}")
        #print(f"Orbital period (P): {P} seconds")
        #print(f"Initial mean anomalies (M0): {M0}")
        #print(f"Initial ascending node longitudes (Omega0): {Omega0}")


        # Mean anomaly (M)
        mean_anomaly = (M0[:, None] + (2 * np.pi / P) * t) % (2 * np.pi)

        # Eccentric anomaly (E)
        eccentric_anom = eccentric_anomaly(e, mean_anomaly)

        # True anomaly (v)
        v = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(eccentric_anom / 2))
        v = np.mod(v, 2 * np.pi)

        # Distance of the satellite to Earth's center (r)
        r = a * (1 - e ** 2) / (1 + e * np.cos(v))

        # True anomaly relative to the line of nodes (gamma)
        gamma = wrap2pi(v + np.radians(ParametersNGSO.omega))  # gamma in the interval [-pi, pi]

        # Latitudes of the satellites, in radians (theta)
        theta = np.arcsin(np.sin(gamma) * np.sin(np.radians(ParametersNGSO.delta)))

        # Longitude variation due to angular displacement, in radians (phiS)
        phiS = np.arccos(np.cos(gamma) / np.cos(theta)) * np.sign(gamma)

        # Longitudes of the ascending node (OmegaG)
        OmegaG = (Omega0[:, None] + EARTH_ROTATION_RATE * t)  # shape (Np*Nsp, len(t))
        OmegaG = wrap2pi(OmegaG)

        # POSITION CALCULATION IN ECEF COORDINATES - ITU-R S.1503

        r_eci = keplerian2eci(a, e, ParametersNGSO.delta, np.degrees(Omega0), ParametersNGSO.omega, np.degrees(v))
        r_ecef = eci2ecef(t, r_eci)
        sx, sy, sz = r_ecef[0], r_ecef[1], r_ecef[2]
        lat = np.degrees(np.arcsin(sz / r))
        lon = np.degrees(np.arctan2(sy, sx))

        num_stations = sx.shape[0]
        num_time_samples = sx.shape[1]

        sat_stations = StationManager(num_stations,num_time_samples)

        data = []
        for i, time_step in enumerate(t):
            idx_stations = 0
            for plane in range(ParametersNGSO.Np):
                for satellite in range(ParametersNGSO.Nsp):
                    sat_index = plane * ParametersNGSO.Nsp + satellite
                    sat_stations.sat_index[idx_stations] = sat_index  # Calculate the index for flattened arrays
                    sat_stations.plane_idx[idx_stations] = plane + 1
                    sat_stations.mean_anomally[idx_stations] = M_o[plane,satellite]
                    sat_stations.long_ascending_node[idx_stations] = Omega_o[plane,satellite]
                    sat_stations.distance = r[sat_index,i] 
                    sat_stations.latitude = lat[sat_index,i]
                    sat_stations.longitude = lon[sat_index,i]
                    sat_stations.x = sx[sat_index,i]
                    sat_stations.y = sy[sat_index,i]
                    sat_stations.height = sz[sat_index,i]

                    # Append data for each satellite at each time step
                    data.append([
                        time_step,  # Time
                        plane + 1,  # Plane number
                        satellite + 1,  # Satellite number
                        M_o[plane, satellite],  # Mean anomaly (M_o) in degrees
                        Omega_o[plane, satellite],  # Longitude of ascending node (Omega_o) in degrees
                        r[sat_index, i],  # distance in km
                        lat[sat_index, i],  # Latitude in degrees
                        lon[sat_index, i],  # Longitude in degrees
                        sx[sat_index, i],  # Cartesian x position
                        sy[sat_index, i],  # Cartesian y position
                        sz[sat_index, i]  # Cartesian z position
                    ])
                    idx_stations = idx_stations + 1

        df = pd.DataFrame(data, columns=['time',
                                     'plane',
                                     'satellite',
                                     'M_o',
                                     'Omega_o',
                                     'r',
                                     'lat',
                                     'lon',
                                     'sx',
                                     'sy',
                                     'sz'])
        sat_stations.dataframe = df


        return sat_stations


