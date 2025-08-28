# -*- coding: utf-8 -*-
"""
Created on wed November  05 15:29:47 2024

@author: https://github.com/joaocabeca2
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
import numpy as np
from multipledispatch import dispatch

from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.station_manager import StationManager


class PropagationP1411(Propagation):
    """
    Implements the propagation general model described in ITU-R P.1411-12, section 4.1.1

    Frequency in MHz and distance in meters!
    """

    def __init__(
            self,
            random_number_gen: np.random.RandomState,
            above_clutter=True):
        super().__init__(random_number_gen)
        self.environment = environment

        if self.environment.upper() == 'URBAN':
            self.alfa = 4.0 
            self.beta = 10.2
            self.gamma = 2.36
            self.sigma = 7.60
        elif self.environment.upper == 'SUBURBAN':
            self.alfa = 5.06
            self.beta = -4.68
            self.gamma = 2.02
            self.sigma = 9.33
        elif self.environment == 'RESIDENTIAL':
            self.alfa = 3.01
            self.beta = 18.8
            self.gamma = 2.07
            self.sigma = 3.07

    
    def get_loss(
        self,
        params: Parameters,
        frequency: float,
        station_a: StationManager,
        station_b: StationManager,
        station_a_gains=None,
        station_b_gains=None,
    ) -> np.array:
        """Wrapper function for the PropagationUMa get_loss method
        Calculates the loss between station_a and station_b

        Parameters
        ----------
        station_a : StationManager
            StationManager container representing IMT UE station - Station_type.IMT_UE
        station_b : StationManager
            StationManager container representing IMT BS stattion
        params : Parameters
            Simulation parameters needed for the propagation class - Station_type.IMT_BS

        Returns
        -------
        np.array
            Return an array station_a.num_stations x station_b.num_stations with the path loss
            between each station
        """
        wrap_around_enabled = False
        if params.imt.topology.type == "MACROCELL":
            wrap_around_enabled = params.imt.topology.macrocell.wrap_around \
                                    and params.imt.topology.macrocell.num_clusters == 1
        if params.imt.topology.type == "HOTSPOT":
            wrap_around_enabled = params.imt.topology.hotspot.wrap_around \
                                    and params.imt.topology.hotspot.num_clusters == 1

        if wrap_around_enabled and (station_a.is_imt_station() and station_b.is_imt_station()):
            distances_2d, distances_3d, _, _ = \
                station_a.get_dist_angles_wrap_around(station_b)
        else:
            self.los_alpha = 2.12
            self.los_beta = 29.2
            self.los_gamma = 2.11
            self.los_sigma = 5.06

            self.nlos_alpha = 4.00
            self.nlos_beta = 10.20
            self.nlos_gamma = 2.36
            self.nlos_sigma = 7.60

    def get_loss(self, *args, **kwargs) -> np.array:
        """Calculate the path loss using the ITU-R P.1411 model.

        Parameters
        ----------
        distance_3D : np.ndarray, optional
            3D distance array between stations (if provided).
        distance_2D : np.ndarray, optional
            2D distance array between stations (if provided).
        frequency : float
            Frequency in Hz.
        los : bool, optional
            If True, use line-of-sight model. Default is True.
        shadow : bool, optional
            If True, include shadow fading. Default is True.
        number_of_sectors : int, optional
            Number of sectors for the calculation. Default is 1.

        Returns
        -------
        np.array
            Path loss values for the given parameters.
        """
        if "distance_3D" in kwargs:
            d = kwargs["distance_3D"]
        else:
            d = kwargs["distance_2D"]

        f = kwargs["frequency"] / 1e3
        los = kwargs.pop("los", True)
        shadow = kwargs.pop("shadow", True)
        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        if los:
            alpha = self.los_alpha
            beta = self.los_beta
            gamma = self.los_gamma
            sigma = self.los_sigma
        else:
            alpha = self.nlos_alpha
            beta = self.nlos_beta
            gamma = self.nlos_gamma
            sigma = self.nlos_sigma

        if shadow:
            shadow_loss = self.random_number_gen.normal(0.0, sigma, d.shape)
        else:
            shadow_loss = 0.0

        loss = 10 * alpha * np.log10(d) + 10 * \
            gamma * np.log10(f) + beta + shadow_loss

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
