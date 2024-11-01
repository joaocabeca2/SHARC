# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:13:33 2018

@author: Calil
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
from sharc.propagation.propagation import Propagation
from multipledispatch import dispatch
from sharc.station_manager import StationManager
from sharc.parameters.parameters import Parameters
import numpy as np


class PropagationP1411_12(Propagation):
    """
    Implements the propagation model described in ITU-R P.1411-12, section 4.2

    Frequency in MHz and distance in meters!
    """

    def __init__(self, random_number_gen: np.random.RandomState, above_rooftop=True, environment=None):
        super().__init__(random_number_gen)

        #Path loss parameters for above-rooftop propagation
        if above_rooftop:
            self.los_alpha = 2.29
            self.los_beta = 28.6
            self.los_gamma = 1.96
            self.los_sigma = 3.48

            self.nlos_alpha = 4.39
            self.nlos_beta = -6.27
            self.nlos_gamma = 2.30
            self.nlos_sigma = 6.89

        #transmission loss coefficients for below-rooftop propagation
        else:
            self.los_alpha = 2.12
            self.los_beta = 29.2
            self.los_gamma = 2.11
            self.los_sigma = 5.06

            self.nlos_alpha = 4.00
            self.nlos_beta = 10.20
            self.nlos_gamma = 2.36
            self.nlos_sigma = 7.60

    def calculate_median_basic_loss(self, distance_3D: np.array,
        frequency: np.array) -> np.array:
        """
        This site-general model is applicable to situations where both the transmitting and receiving stations 
        are located below-rooftop, regardless of their antenna heights.

        Parameters
        ----------
        distance_3D : np.array
            3D direct distance between the transmitting and receiving stations (in meters).
        frequency : np.array
            Operating frequency in GHz.
        
        Returns
        -------
        np.array
            Median basic transmission loss in dB.
        """
        median_loss = 10 * self.los_alfa * np.log10(distance_3D) + self.los_beta + 10 * self.los_gamma * np.log10(frequency)
        # Add zero-mean Gaussian random variable for shadowing
        shadowing = self.random_number_gen.normal(0, self.los_sigma, distance_3D.shape)

        return median_loss + shadowing

    def calculate_free_space_loss(self, distance_3D: np.array, frequency: np.array) -> np.array:
        speed_light = 3e8 #m/s
        """
        Calculates the free-space basic transmission loss, LFS.

        Parameters
        ----------
        distance_3D : np.array
            3D direct distance between transmitting and receiving stations (m).
        frequency : np.array
            Operating frequency (GHz).

        Returns
        -------
        np.array
            Free-space basic transmission loss in dB.
        """
        # Convert frequency to Hz from GHz
        frequency *= 1e9
        # Calculate LFS using the given formula
        return 20 * np.log10(4 * 10**9 * np.pi * distance_3D * frequency /speed_light)


    def calculate_excess_loss(self, lfs: np.array, median_basic_loss: np.array, distance_3D: np.array) -> np.array:
        """
        Calculates the excess basic transmission loss for NLoS scenarios.
        
        This method computes the excess loss with respect to the free-space basic transmission loss
        using a Monte Carlo approach. It uses a random variable A, which is normally distributed
        with mean μ and standard deviation σ, to account for the variability in the signal propagation
        environment.

        Parameters
        ----------
        lfs : np.array
            Free-space basic transmission loss, LFS, in dB.
        median_basic_loss : np.array
            Median basic transmission loss, Lb(d, f), in dB.
        distance_3D : np.array
            3D direct distance between the transmitting and receiving stations (in meters).

        Returns
        -------
        np.array
            Excess basic transmission loss in dB.
        """
        # Calculate μ as the difference between the median basic loss and free-space loss
        u = median_basic_loss - lfs

        # Generate the random variable A using a normal distribution with mean μ and standard deviation σ
        a = self.random_number_gen.normal(u, self.los_sigma, distance_3D.shape)

        # Compute the excess loss using the specified formula
        return 10 * np.log10(10**(0.1 * a) + 1)


    @dispatch(Parameters, float, StationManager, StationManager, np.ndarray, np.ndarray)
    def get_loss(
        self,
        params: Parameters,
        frequency: float,
        station_a: StationManager,
        station_b: StationManager,
        station_a_gains=None,
        station_b_gains=None,
    ) -> np.array:
        pass

    def get_loss_nlos(
        self, distance_2D: np.array, distance_3D: np.array,
        frequency: np.array,
        h_bs: np.array, h_ue: np.array, h_e: np.array,
        shadowing_std=6,
    ):
        pass

if __name__ == '__main__':
    pass