import math 
import numpy as np  
import matplotlib.pyplot as plt

from sharc.topology.topology_ntn_EESS import TopologyMacrocell
from sharc.antenna.antenna_s1528 import AntennaS1528, AntennaS1528Leo
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.station_factory import StationFactory
from sharc.parameters.parameters_imt import ParametersImt
from sharc.parameters.parameters_mss_ss import ParametersMssSs
from sharc.parameters.parameters_eess_passive import ParametersEessPassive
from sharc.parameters.parameters_antenna_imt import ParametersAntennaImt
from sharc.parameters.parameters_antenna_s1528 import ParametersAntennaS1528
from sharc.mask.spectral_mask_imt import SpectralMaskImt

def generate_eess(param_mss: ParametersMssSs, param_eess: ParametersEessPassive):
    ntn_topology = TopologyMacrocell(
    )