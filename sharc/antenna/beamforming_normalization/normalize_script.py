# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:26:45 2018

@author: Calil

This script generates the correction factors for the IMT Beamforming Antennas,
both array and single element, and saves them in files with the given names.
This script must be ran with the appropriate parameters prior to using any
normalization in the SHARC simulator, since the simulator merely reads the
correction factor values from the saved files.
For the co-channel scenario (antenna array) the correction factor is a 2-D
array with the lines representing the azimuth and the columns representing the
elevation of the beam direction.
For the adjacent channel scenario (single element) the correction factor is a
float.

Variables:
    resolution (float): resolution of the azimuth and elevation angles in the
        antenna array correction factor matrix [deg]. This defines the number
        of beam pointing directions to which the correction factor is
        calculated.
    tolerance (float): absolute tolerance of the correction factor integral, in
        linear scale.
    norm (BeamformingNormalizer): object that calculates the normalization.
    param_list (list): list of antenna parameters to which calculate the
        correction factors. New parameters are added as:
            ParametersAntennaImt(adjacent_antenna_model,
                       normalization,
                       norm_data,
                       element_pattern,
                       element_max_g,
                       element_phi_deg_3db,
                       element_theta_deg_3db,
                       element_am,
                       element_sla_v,
                       n_rows,
                       n_columns,
                       element_horiz_spacing,
                       element_vert_spacing,
                       minimum_array_gain,
                       downtilt_deg)
            normalization parameter must be set to False, otherwise script will
            try to normalize an already normalized antenna.
    file_names (list): list of file names to which save the normalization data.
        Files are paired with ParametersAntennaImt objects in param_list, so that the
        normalization data of the first element of param_list is saved in a
        file with the name specified in the first element of file_names and so
        on.

Data is saved in an .npz file in a dict like data structure with the
following keys:
    resolution (float): antenna array correction factor matrix angle resolution
        [deg]
    phi_range (tuple): range of beam pointing azimuth angle values [deg]
    theta_range (tuple): range of beam pointing elevation angle values [deg]
    correction_factor_co_channel (2D np.array): correction factor [dB]for the
        co-channel scenario (antenna array) for each of the phi theta pairs in
        phi_range and theta_range. Phi is associated with the lines and Theta
        is associated with the columns of the array.
    error_co_channel (2D np.array of tuples): lower and upper bounds of
        calculated correction factors [dB], considering integral error
    correction_factor_adj_channel (float):correction factor for single antenna
        element
    error_adj_channel (tuple): lower and upper bounds [dB] of single antenna
        element correction factor
    parameters (ParametersAntennaImt): antenna parameters used in the normalization
"""

from sharc.antenna.beamforming_normalization.beamforming_normalizer import BeamformingNormalizer
from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


if __name__ == "__main__":

    ###########################################################################
    # List of antenna parameters to which calculate the normalization factors.
    file_names = ["bs_norm_8x8_050.npz"]
    param_list = [
        ParametersAntennaImt(
            # not needed here
            adjacent_antenna_model="",
            # not needed here
            normalization=False,
            # not needed here
            normalization_file=None,
            element_pattern="M2101",
            element_max_g=5,
            element_phi_3db=65,
            element_theta_3db=65,
            element_am=30,
            element_sla_v=30,
            n_rows=8,
            n_columns=8,
            element_horiz_spacing=0.5,
            element_vert_spacing=0.5,
            multiplication_factor=12,
            minimum_array_gain=-200,
            downtilt=0,
        ),
    ]
    ###########################################################################
    # Setup
    # General parameters
    resolution = 5
    tolerance = 1e-2

    # Create object
    norm = BeamformingNormalizer(resolution, tolerance)
    ###########################################################################
    # Normalize and save
    for par, file in zip(param_list, file_names):
        s = 'Generating ' + file
        print(s)

        norm.generate_correction_matrix(par, file)
