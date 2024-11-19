# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:29:48 2017

@author: edgar
"""

import numpy as np

from sharc.support.enumerations import StationType
from sharc.station import Station
from sharc.antenna.antenna import Antenna
from sharc.mask.spectral_mask_3gpp import SpectralMask3Gpp
import pandas as pd


class StationManager(object):
    """
    This is the base class that manages an array of stations that will be
    used during a simulation. It acts like a container that vectorizes the
    station properties to speed up calculations.
    """

    def __init__(self, n, m):
        self.num_stations = n
        self.num_time_samples = m
        self.sat_index = np.empty(n)
        self.plane_idx = np.empty(n)
        self.sat_number = np.empty(n)
        self.mean_anomally = np.empty(n)
        self.long_ascending_node = np.empty(n)
        
        # Multidimensional arrays
        self.x = np.empty((n, m))
        self.y = np.empty((n, m))
        self.azimuth = np.empty((n, m))
        self.elevation = np.empty((n, m))
        self.distance = np.empty((n, m))
        self.latitude = np.empty((n, m))
        self.longitude = np.empty((n, m))
        self.height = np.empty((n, m))
        self.indoor = np.zeros((n, m), dtype=bool)
        self.active = np.ones((n, m), dtype=bool)
        self.tx_power = np.empty((n, m))
        self.rx_power = np.empty((n, m))
        self.rx_interference = np.empty((n, m))  # Rx interference in dBW
        self.ext_interference = np.empty((n, m))  # External interference in dBW
        self.snr = np.empty((n, m))
        self.sinr = np.empty((n, m))
        self.sinr_ext = np.empty((n, m))
        self.inr = np.empty((n, m))  # INR in dBm/MHz
        self.pfd = np.empty((n, m))  # Powerflux density in dBm/m^2

        # Single-dimensional arrays with specific dtypes
        self.antenna = np.empty(n, dtype=object)  # Array of Antenna objects
        self.bandwidth = np.empty(n)  # Bandwidth in MHz
        self.noise_figure = np.empty(n)
        self.noise_temperature = np.empty(n)
        self.thermal_noise = np.empty(n)
        self.total_interference = np.empty(n)
        self.center_freq = np.empty(n)
        self.spectral_mask = np.empty(n, dtype=object)  # Array of SpectralMask3Gpp objects

        # Other attributes
        self.station_type = StationType.NONE
        self.is_space_station = False
        self.intersite_dist = 0.0
        self.dataframe = pd.DataFrame()
