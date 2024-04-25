# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:35:52 2017

@author: edgar
"""

import sys
import os
import configparser

from sharc.parameters.parameters_general import ParametersGeneral
from sharc.parameters.parameters_imt import ParametersImt
from sharc.parameters.parameters_hotspot import ParametersHotspot
from sharc.parameters.parameters_indoor import ParametersIndoor
from sharc.parameters.parameters_antenna_imt import ParametersAntennaImt
from sharc.parameters.parameters_eess_passive import ParametersEessPassive
from sharc.parameters.parameters_fs import ParametersFs
from sharc.parameters.parameters_fss_ss import ParametersFssSs
from sharc.parameters.parameters_fss_es import ParametersFssEs
from sharc.parameters.parameters_haps import ParametersHaps
from sharc.parameters.parameters_rns import ParametersRns
from sharc.parameters.parameters_ras import ParametersRas


class Parameters(object):
    """
    Reads parameters from input file.
    """

    def __init__(self):
        self.file_name = None

        self.general = ParametersGeneral()
        self.imt = ParametersImt()
        self.antenna_imt = ParametersAntennaImt()
        self.hotspot = ParametersHotspot()
        self.indoor = ParametersIndoor()
        self.eess_passive = ParametersEessPassive()
        self.fs = ParametersFs()
        self.fss_ss = ParametersFssSs()
        self.fss_es = ParametersFssEs()
        self.haps = ParametersHaps()
        self.rns = ParametersRns()
        self.ras = ParametersRas()


    def set_file_name(self, file_name: str):
        """sets the configuration file name

        Parameters
        ----------
        file_name : str
            configuration file path
        """
        if not os.path.isfile(file_name):
            err_msg = f"PARAMETER ERROR [{self.__class__.__name__}]: \
                Could not find the configuration file {file_name}"
            sys.stderr.write(err_msg)
            sys.exit(1)
        self.file_name = file_name


    def read_params(self):
        """Read the parameters from the config file
        """
        config = configparser.ConfigParser()
        config.read(self.file_name)

        #######################################################################
        # GENERAL
        #######################################################################
        self.general.load_parameters_from_file(self.file_name)

        #######################################################################
        # IMT
        #######################################################################
        self.imt.load_parameters_from_file(self.file_name)

        #######################################################################
        # IMT ANTENNA
        #######################################################################
        self.antenna_imt.adjacent_antenna_model     = config.get("IMT_ANTENNA", "adjacent_antenna_model")
        self.antenna_imt.bs_normalization           = config.getboolean("IMT_ANTENNA", "bs_normalization")
        self.antenna_imt.ue_normalization           = config.getboolean("IMT_ANTENNA", "ue_normalization")
        self.antenna_imt.bs_normalization_file      = config.get("IMT_ANTENNA", "bs_normalization_file")
        self.antenna_imt.ue_normalization_file      = config.get("IMT_ANTENNA", "ue_normalization_file")
        self.antenna_imt.bs_element_pattern         = config.get("IMT_ANTENNA", "bs_element_pattern")
        self.antenna_imt.ue_element_pattern         = config.get("IMT_ANTENNA", "ue_element_pattern")
        
        self.antenna_imt.bs_element_max_g           = config.getfloat("IMT_ANTENNA", "bs_element_max_g")
        self.antenna_imt.bs_element_phi_3db         = config.getfloat("IMT_ANTENNA", "bs_element_phi_3db")
        self.antenna_imt.bs_element_theta_3db       = config.getfloat("IMT_ANTENNA", "bs_element_theta_3db")
        self.antenna_imt.bs_element_am              = config.getfloat("IMT_ANTENNA", "bs_element_am")
        self.antenna_imt.bs_element_sla_v           = config.getfloat("IMT_ANTENNA", "bs_element_sla_v")
        self.antenna_imt.bs_n_rows                  = config.getfloat("IMT_ANTENNA", "bs_n_rows")
        self.antenna_imt.bs_n_columns               = config.getfloat("IMT_ANTENNA", "bs_n_columns")
        self.antenna_imt.bs_element_horiz_spacing   = config.getfloat("IMT_ANTENNA", "bs_element_horiz_spacing")
        self.antenna_imt.bs_element_vert_spacing    = config.getfloat("IMT_ANTENNA", "bs_element_vert_spacing")
        self.antenna_imt.bs_multiplication_factor   = config.getfloat("IMT_ANTENNA", "bs_multiplication_factor")
        self.antenna_imt.bs_minimum_array_gain      = config.getfloat("IMT_ANTENNA", "bs_minimum_array_gain")
        
        self.antenna_imt.ue_element_max_g           = config.getfloat("IMT_ANTENNA", "ue_element_max_g")
        self.antenna_imt.ue_element_phi_3db         = config.getfloat("IMT_ANTENNA", "ue_element_phi_3db")
        self.antenna_imt.ue_element_theta_3db       = config.getfloat("IMT_ANTENNA", "ue_element_theta_3db")
        self.antenna_imt.ue_element_am              = config.getfloat("IMT_ANTENNA", "ue_element_am")
        self.antenna_imt.ue_element_sla_v           = config.getfloat("IMT_ANTENNA", "ue_element_sla_v")
        self.antenna_imt.ue_n_rows                  = config.getfloat("IMT_ANTENNA", "ue_n_rows")
        self.antenna_imt.ue_n_columns               = config.getfloat("IMT_ANTENNA", "ue_n_columns")
        self.antenna_imt.ue_element_horiz_spacing   = config.getfloat("IMT_ANTENNA", "ue_element_horiz_spacing")
        self.antenna_imt.ue_element_vert_spacing    = config.getfloat("IMT_ANTENNA", "ue_element_vert_spacing")
        self.antenna_imt.ue_multiplication_factor   = config.getfloat("IMT_ANTENNA", "ue_multiplication_factor")
        self.antenna_imt.ue_minimum_array_gain      = config.getfloat("IMT_ANTENNA", "ue_minimum_array_gain")

        self.antenna_imt.bs_downtilt            = config.getfloat("IMT_ANTENNA", "bs_downtilt")

        #######################################################################
        # HOTSPOT
        #######################################################################
        self.hotspot.num_hotspots_per_cell = config.getint("HOTSPOT", "num_hotspots_per_cell")
        self.hotspot.max_dist_hotspot_ue   = config.getfloat("HOTSPOT", "max_dist_hotspot_ue")
        self.hotspot.min_dist_bs_hotspot   = config.getfloat("HOTSPOT", "min_dist_bs_hotspot")

        #######################################################################
        # INDOOR
        #######################################################################
        self.indoor.basic_path_loss = config.get("INDOOR", "basic_path_loss")
        self.indoor.n_rows = config.getint("INDOOR", "n_rows")
        self.indoor.n_colums = config.getint("INDOOR", "n_colums")
        self.indoor.num_imt_buildings = config.get("INDOOR", "num_imt_buildings")
        self.indoor.street_width = config.getint("INDOOR", "street_width")
        self.indoor.intersite_distance = config.getfloat("INDOOR", "intersite_distance")
        self.indoor.num_cells = config.getint("INDOOR", "num_cells")
        self.indoor.num_floors = config.getint("INDOOR", "num_floors")
        self.indoor.ue_indoor_percent = config.getfloat("INDOOR", "ue_indoor_percent")
        self.indoor.building_class = config.get("INDOOR", "building_class")

        #######################################################################
        # FSS space station
        #######################################################################
        self.fss_ss.load_parameters_from_file(self.file_name)

        #######################################################################
        # FSS earth station
        #######################################################################
        self.fss_es.load_parameters_from_file(self.file_name)

        #######################################################################
        # Fixed wireless service
        #######################################################################
        self.fs.load_parameters_from_file(self.file_name)

        #######################################################################
        # HAPS (airbone) station
        #######################################################################
        self.haps.load_parameters_from_file(self.file_name)

        #######################################################################
        # RNS
        #######################################################################
        self.rns.load_parameters_from_file(self.file_name)

        #######################################################################
        # RAS station
        #######################################################################
        self.ras.load_parameters_from_file(self.file_name)

        #######################################################################
        # EESS passive
        #######################################################################
        self.eess_passive.frequency               = config.getfloat("EESS_PASSIVE", "frequency")
        self.eess_passive.bandwidth               = config.getfloat("EESS_PASSIVE", "bandwidth")
        self.eess_passive.nadir_angle             = config.getfloat("EESS_PASSIVE", "nadir_angle")
        self.eess_passive.altitude                = config.getfloat("EESS_PASSIVE", "altitude")
        self.eess_passive.antenna_pattern         = config.get("EESS_PASSIVE", "antenna_pattern")
        self.eess_passive.antenna_efficiency      = config.getfloat("EESS_PASSIVE", "antenna_efficiency")
        self.eess_passive.antenna_diameter        = config.getfloat("EESS_PASSIVE", "antenna_diameter")
        self.eess_passive.antenna_gain            = config.getfloat("EESS_PASSIVE", "antenna_gain")
        self.eess_passive.channel_model           = config.get("EESS_PASSIVE", "channel_model")
        self.eess_passive.imt_altitude            = config.getfloat("EESS_PASSIVE", "imt_altitude")
        self.eess_passive.imt_lat_deg             = config.getfloat("EESS_PASSIVE", "imt_lat_deg")
        self.eess_passive.season                  = config.get("EESS_PASSIVE", "season")
        self.eess_passive.BOLTZMANN_CONSTANT      = config.getfloat("EESS_PASSIVE", "BOLTZMANN_CONSTANT")
        self.eess_passive.EARTH_RADIUS            = config.getfloat("EESS_PASSIVE", "EARTH_RADIUS")
