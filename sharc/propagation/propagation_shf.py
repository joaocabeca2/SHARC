# -*- coding: utf-8 -*-
"""
Created on wed November  05 15:29:47 2024

@author: joaocabeca2
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
import numpy as np

from sharc.parameters.constants import SPEED_OF_LIGHT
from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.station_manager import StationManager


class PropagationSHF(Propagation):
    """
    Implements the Super High Frequency model described in ITU-R P.1411-12, section 4.1.2

    Frequency in MHz and distance in meters!
    """

    def __init__(self, 
                random_number_gen: np.random.RandomState,
                road_height: float
        ):
        super().__init__(random_number_gen)
        self.road_height = road_height
  
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
        
        return self._get_loss(distance_3D,
                            distance_2D,
                            frequency * np.ones(distance_2D.shape),
                            station_b.height,
                            station_a.height,
                            )


        
    def _get_loss(
        self, distance_3d: np.array, distance_2d: np.array, frequency: np.array,
        bs_height: np.array, ue_height: np.array
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
        wavelenght = SPEED_OF_LIGHT / frequency
        self.road_height *= np.ones(distance_2D.shape)
        #for path lengths up to about 1 km, road traffic will influence the effective road height and 
        #will thus affect the breakpoint distance. This distance is estimated by:
        breakpoint_dist = 4 * ((h_bs - self.road_height) * (h_ue - self.road_height)) / wavelenght

        if (ue_height > self.road_height).all() and (bs_height > self.road_height).all():
            
            basic_loss = 20 * (np.log10(wavelenght**2 / (8 * np.pi * (h_bs - self.road_height) * (h_ue - self.road_height))))

            lower_bound_loss, upper_bound_loss = self.calculate_bound_loss(basic_loss, distance_3D, breakpoint_dist)

            return basic_loss, lower_bound_loss, upper_bound_loss
        
        else:
            #Case breakpoint doesnt exist
            rs = 20
            if distance_3D < rs:
                basic_loss = 20 * (np.log10(wavelenght**2 / (8 * np.pi * h_bs * h_ue)))
                lower_bound_loss, upper_bound_loss = self.calculate_bound_loss(basic_loss, distance_3D, breakpoint_dist)
            
            else:
                basic_loss = 20 * (np.log10(wavelenght / (2 * np.pi * rs)))
                lower_bound_loss = basic_loss + (30 * np.log10(distance_3D / rs))
                upper_bound_loss = basic_loss + 20 + (30 * np.log10(distance_3D / rs))
                #median_bound_loss = basic_loss + 6 + 30 * np.log10(distance_3D)

            return basic_loss, lower_bound_loss, upper_bound_loss
            
    
    def calculate_bound_loss(self, basic_loss: np.array, distance_3d: np.array, breakpoint_dist: float):
        # Apply distance-dependent factors
        lower_bound_loss = np.where(
            distance_3d <= breakpoint_dist,
            basic_loss + (20 * np.log10(distance_3d / breakpoint_dist)),
            basic_loss + (40 * np.log10(distance_3d / breakpoint_dist))
        )

        upper_bound_loss = np.where(
            distance_3d <= breakpoint_dist,
            basic_loss + 20 + (25 * np.log10(distance_3d / breakpoint_dist)),
            basic_loss + 20 + (40 * np.log10(distance_3d / breakpoint_dist))
        )

        return lower_bound_loss, upper_bound_loss


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Configuração de parâmetros
    num_ue = 1
    num_bs = 1000
    h_bs = 4 * np.ones(num_bs)
    h_ue = 2.7 * np.ones(num_ue)

    # Configuração da distância para o cenário
    distance_2D = np.repeat(np.linspace(1, 1000, num=num_bs)[np.newaxis, :], num_ue, axis=0)
    frequency = 7 * np.ones(num_bs)  
    distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[np.newaxis, :])**2)

    # Gerador de números aleatórios
    random_number_gen = np.random.RandomState(101)

    shf = PropagationSHF(random_number_gen, 1.6)
    #free_space_prop = PropagationFreeSpace(random_number_gen)

    #free_space_loss = free_space_prop.get_free_space_loss(frequency * 1000, distance_3D)
    basic_loss, lower_bound_loss, upper_bound_loss = shf._get_loss(distance_3D,distance_2D,frequency,h_bs,h_ue)

    # Plotando os gráficos
    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
    ax = fig.gca()
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b']))

    ax.semilogx(distance_2D[0, :], upper_bound_loss[0, :], label="Upper Bound Loss")
    ax.semilogx(distance_2D[0, :], basic_loss[0, :], label="Basic Loss")
    ax.semilogx(distance_2D[0, :], lower_bound_loss[0, :], label="Lower Bound Loss")

    plt.title("SHF - Propagation")
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.xlim((0, distance_2D[0, -1]))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid()

    plt.show()