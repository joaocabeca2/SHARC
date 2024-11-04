# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:29:47 2017

@author: LeticiaValle_Mac
"""
import numpy as np
from multipledispatch import dispatch

from sharc.propagation.propagation import Propagation
from sharc.station_manager import StationManager
from sharc.parameters.parameters import Parameters


class PropagationUHF(Propagation):
    """
    Implements the Urban Micro path loss model (Street Canyon) with LOS
    probability according to 3GPP TR 38.900 v14.2.0.
    TODO: calculate the effective environment height for the generic case
    """

    def __init__(
        self,
        random_number_gen: np.random.RandomState,
        los_adjustment_factor: float,
    ):
        super().__init__(random_number_gen)
        self.los_adjustment_factor = los_adjustment_factor
        self.c = 3e8

        
    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool)
    def get_loss(
        self,
        distance_3D: np.array,
        distance_2D: np.array,
        frequency: np.array,
        bs_height: np.array,
        ue_height: np.array,
        shadowing_flag: bool,
        loss : np.array,
    ) -> np.array:
        """
        Calculates path loss for LOS and NLOS cases with respective shadowing
        (if shadowing is to be added)

        Parameters
        ----------
            distance_3D (np.array) : 3D distances between base stations and user equipment
            distance_2D (np.array) : 2D distances between base stations and user equipment
            frequency (np.array) : center frequencies [MHz]
            bs_height (np.array) : base station antenna heights
            ue_height (np.array) : user equipment antenna heights
            shadowing (bool) : if shadowing should be added or not

        Returns
        -------
            array with path loss values with dimensions of distance_2D
        """

        wavelenght = self.c/frequency
        breakpoint_dist = (4*bs_height*ue_height) / wavelenght

        if breakpoint_dist <= distance_3D:
            lower_bound_aproximation = loss + (20 * np.log10(distance_3D / breakpoint_dist))
        else:
            lower_bound_aproximation = loss + (40 * np.log10(distance_3D / breakpoint_dist))
        
        
        
