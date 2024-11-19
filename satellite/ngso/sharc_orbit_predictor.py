import numpy as np
import pandas as pd
from scipy.special import jv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from parametetsNGSO import ParametersNGSO
from custom_functions import wrap2pi, eccentric_anomaly, keplerian2eci, eci2ecef
from constants import  EARTH_RADIUS_KM, KEPLER_CONST, EARTH_ROTATION_RATE
from station_factory import StationFactory
# START OF TIMER
start_time = time.time()



# INPUTS - NGSO SYSTEM

# Globalstar
globalStar = ParametersNGSO()

time_interval = 5
number_orbital_simulations = 4

sat_sation = StationFactory.generate_satellite_station(globalStar,time_interval,number_orbital_simulations)

print(sat_sation.dataframe)

