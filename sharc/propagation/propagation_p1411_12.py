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


class PropagationP1411_12(Propagation):
    """
    Implements the propagation model described in ITU-R P.1411-12, section 4.2

    Frequency in MHz and distance in meters!
    """

    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)
    
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
            distances_2d = station_a.get_distance_to(station_b)
            distances_3d = station_a.get_3d_distance_to(station_b)

        '''median_basic_loss = self.calculate_median_basic_loss(
            distances_3d,
            frequency * np.ones(distances_2d.shape)
        )

        free_space_prop = PropagationFreeSpace()
        free_space_loss = free_space_prop.get_free_space_loss(frequency * np.ones(distance_2D.shape), distance_3D)
        excess_loss = self.calculate_excess_loss(free_space_loss, median_basic_loss, distance_3D)

        # the interface expects station_a.num_stations x station_b.num_stations array
        return median_basic_loss'''

    def calculate_median_basic_loss(self, distance_3D: np.array,
        frequency: np.array,
        los_alfa: np.array,
        los_beta: np.array,
        los_gamma: np.array,
        los_sigma: np.array,
        random_number_gen: np.random.Generator
    ) -> np.array:
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
        median_loss = (10 * los_alfa * np.log10(distance_3D)) + los_beta + (10 * los_gamma * np.log10(frequency))
        # Add zero-mean Gaussian random variable for shadowing
        shadowing = random_number_gen.normal(0, los_sigma, distance_3D.shape)

        return median_loss + shadowing


    def calculate_excess_loss(self, lfs: np.array, median_basic_loss: np.array, distance_3D: np.array, los_sigma: np.array) -> np.array:
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
        a = self.random_number_gen.normal(u, los_sigma, distance_3D.shape)

        # Compute the excess loss using the specified formula
        return 10 * np.log10(10**(0.1 * a) + 1)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Configuração de parâmetros
    num_ue = 1
    num_bs = 1000
    h_bs = 6 * np.ones(num_bs)
    h_ue = 1.5 * np.ones(num_ue)

    # situations to be used for below-rooftop propagation in urban and suburban environments provided in ITU.R P1411-2 sec 4.1
    scenarios = [
    {"name": "Urban low-rise/Suburban (LoS)", "alfa": 2.12, "beta": 29.2, "gamma": 2.11, "sigma": 5.06, "distance_range": (5, 660), "frequency": (800, 82000)},
    {"name": "Urban high-rise (NLoS)", "alfa": 4.00, "beta": 10.2, "gamma": 2.36, "sigma": 7.60, "distance_range": (30, 715), "frequency": (800, 82000)},
    {"name": "Urban low-rise/Suburban (NLoS)", "alfa": 5.06, "beta": -4.68, "gamma": 2.02, "sigma": 9.33, "distance_range": (30, 250), "frequency": (1000, 73000)},
    {"name": "Residential (NLoS)", "alfa": 3.01, "beta": 18.8, "gamma": 2.07, "sigma": 3.07, "distance_range": (30, 170), "frequency": (800, 73000)},
]
    for scenario in scenarios:
        # Parâmetros específicos do cenário
        los_alfa = scenario["alfa"] * np.ones(num_bs)
        los_beta = scenario["beta"] * np.ones(num_bs)
        los_gamma = scenario["gamma"] * np.ones(num_bs)
        los_sigma = scenario["sigma"] * np.ones(num_bs)
    
        # Configuração da distância para o cenário
        distance_2D = np.repeat(np.linspace(scenario["distance_range"][0], scenario["distance_range"][1], num=num_bs)[np.newaxis, :], num_ue, axis=0)
        frequency = np.linspace(scenario["frequency"][0], scenario["frequency"][1], num=num_bs)   
        distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[np.newaxis, :])**2)

        # Gerador de números aleatórios
        random_number_gen = np.random.RandomState(101)

        p1411 = PropagationP1411_12(random_number_gen)
        free_space_prop = PropagationFreeSpace(random_number_gen)

        free_space_loss = free_space_prop.get_free_space_loss(frequency, distance_3D)
        median_basic_loss = p1411.calculate_median_basic_loss(distance_3D, frequency, los_alfa,
                                                            los_beta, los_gamma, los_sigma, random_number_gen)
        excess_loss = p1411.calculate_excess_loss(free_space_loss, median_basic_loss, distance_3D, los_sigma)

        # Plotando os gráficos
        fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
        ax = fig.gca()
        ax.set_prop_cycle(cycler('color', ['r', 'g', 'b']))

        ax.semilogx(distance_2D[0, :], median_basic_loss[0, :], label="Median basic Loss")
        ax.semilogx(distance_2D[0, :], excess_loss[0, :], label="Excess Loss")
        ax.semilogx(distance_2D[0, :], free_space_loss[0, :], label="Free space Loss")

        plt.title(scenario["name"])
        plt.xlabel("Distance [m]")
        plt.ylabel("Path Loss [dB]")
        plt.xlim((0, distance_2D[0, -1]))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.grid()

        plt.show()