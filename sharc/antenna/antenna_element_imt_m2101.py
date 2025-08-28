# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:13:58 2017

@author: Calil
"""

import numpy as np

from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt


class AntennaElementImtM2101(object):
    """
    Implements a single element of an IMT antenna array.

    Attributes
    ----------
        g_max (float): maximum gain of element
        theta_3db (float): vertical 3dB beamwidth of single element [degrees]
        phi_3db (float): horizontal 3dB beamwidth of single element [degrees]
        am (float): front-to-back ratio
        sla_v (float): element vertical sidelobe attenuation
    """

    def __init__(self, par: ParametersAntennaImt):
        """
        Constructs an AntennaElementImt object.

        Parameters
        ---------
            param (ParametersAntennaImt): antenna IMT parameters
        """
        self.param = par

        self.multiplication_factor = par.multiplication_factor
        self.g_max = par.element_max_g
        self.phi_3db = par.element_phi_3db
        self.theta_3db = par.element_theta_3db
        self.am = par.element_am
        self.sla_v = par.element_sla_v

    def horizontal_pattern(self, phi: np.array) -> np.array:
        """
        Calculates the horizontal radiation pattern.

        Parameters
        ----------
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            a_h (np.array): horizontal radiation pattern gain value
        """
        return -1.0 * np.minimum(self.multiplication_factor *
                                 (phi / self.phi_3db)**2, self.am)

    def vertical_pattern(self, theta: np.array) -> np.array:
        """
        Calculates the vertical radiation pattern.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]

        Returns
        -------
            a_v (np.array): vertical radiation pattern gain value
        """
        return -1.0 * np.minimum(self.multiplication_factor *
                                 ((theta - 90.0) / self.theta_3db)**2, self.sla_v)

    def element_pattern(self, phi: np.array, theta: np.array) -> np.array:
        """
        Calculates the element radiation pattern gain.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            gain (np.array): element radiation pattern gain value [dBi]
        """
        att = -1.0 * (
            self.horizontal_pattern(phi) +
            self.vertical_pattern(theta)
        )
        gain = self.g_max - np.minimum(att, self.am)

        return gain
