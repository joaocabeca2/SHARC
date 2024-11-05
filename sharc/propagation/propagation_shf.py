# -*- coding: utf-8 -*-
"""
Created on wed November  05 15:29:47 2024

@author: https://github.com/joaocabeca2
"""
import numpy as np
from multipledispatch import dispatch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.station_manager import StationManager


class PropagationSHF(Propagation):
    """
    Represents the propagation characteristics in the Ultra High Frequency (UHF) range.

    In the UHF frequency range, basic transmission loss can be characterized by two slopes
    and a single breakpoint, as defined by Recommendation ITU-R P.1411-12
    """
    

    def __init__(
        self,
        random_number_gen: np.random.RandomState
    ):
        super().__init__(random_number_gen)

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
        """Wrapper function for the PropagationUHF get_loss method
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
            distance_2d, distance_3d, _, _ = \
                station_a.get_dist_angles_wrap_around(station_b)
        else:
            distance_2d = station_a.get_distance_to(station_b)
            distance_3d = station_a.get_3d_distance_to(station_b)

        loss = self.get_loss(
            distance_3d,
            distance_2d,
            frequency * np.ones(distance_2d.shape),
            station_b.height,
            station_a.height,
            params.imt.shadowing,
        )

        return loss
    
    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool)
    def get_loss(
        self,
        distance_3D: np.array,
        frequency: np.array,
        bs_height: np.array,
        ue_height: np.array,
        road_height : np.array,
        heavy_traffic: bool,
    ) -> np.array:

        c = 3e8  # velocidade da luz
        wavelength = c / frequency  # comprimento de onda
        
        # Cálculo do breakpoint_dist
        if heavy_traffic:
            breakpoint_dist = (4 * ((bs_height - road_height) * (ue_height - road_height))) / wavelength
        else:
            breakpoint_dist = 0
        
        # Condições para alturas da estação base (BS) e da unidade do usuário (UE)
        if np.all(ue_height > road_height) and np.all(bs_height > road_height):
            # Perda básica de transmissão para ambos acima da altura da estrada
            basic_transmission_loss = 20 * np.log10(wavelength**2 / (8 * np.pi * (bs_height - road_height) * (ue_height - road_height)))
            
            # Limites de perda de transmissão
            lower_bound_loss = np.where(
                breakpoint_dist >= distance_3D,
                basic_transmission_loss + (20 * np.log10(distance_3D / breakpoint_dist)),
                basic_transmission_loss + (40 * np.log10(distance_3D / breakpoint_dist))
            )

            upper_bound_loss = np.where(
                breakpoint_dist >= distance_3D,
                basic_transmission_loss + 20 + (25 * np.log10(distance_3D / breakpoint_dist)),
                basic_transmission_loss + 20 + (40 * np.log10(distance_3D / breakpoint_dist))
            )

            # Como ambos estão acima da altura da estrada, `median_bound_loss` não é calculado
            median_bound_loss = np.full(lower_bound_loss.shape, np.nan)

        else:
            # Perda básica de transmissão para quando um ou ambos estão abaixo da altura da estrada
            basic_transmission_loss = 20 * np.log10(wavelength / (2 * np.pi * road_height))
            
            # Limites de perda de transmissão para esta condição
            lower_bound_loss = basic_transmission_loss + (30 * np.log10(distance_3D / road_height))
            upper_bound_loss = basic_transmission_loss + 20 + (30 * np.log10(distance_3D / road_height))
            median_bound_loss = basic_transmission_loss + 6 + (30 * np.log10(distance_3D / road_height))
        
        return np.array([lower_bound_loss, upper_bound_loss, median_bound_loss])

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from cycler import cycler

    # Configuração de parâmetros
    num_ue = 1
    num_bs = 1000
    distance_2D = np.repeat(np.linspace(1, num_bs, num=num_bs)[np.newaxis, :], num_ue, axis=0)
    freq = 27000 * np.ones(distance_2D.shape)
    h_bs = 6 * np.ones(num_bs)
    h_ue = 1.5 * np.ones(num_ue)  # Altura do usuário
    h_road = 1 * np.ones(num_bs)  # Altura efetiva da estrada

    # Cálculo da distância 3D
    distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[np.newaxis, :])**2)

    # Criação do objeto PropagationSHF e cálculo da perda
    shf = PropagationSHF(np.random.RandomState(101))
    loss = shf.get_loss(distance_3D, freq, h_bs, h_ue, h_road, True)  # Removendo h_road se não for necessário

    # Separando os valores de perda
    lower_bound_loss, upper_bound_loss, median_bound_loss = loss

    # Plotando os gráficos
    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
    ax = fig.gca()
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b']))

    ax.semilogx(distance_2D[0, :], lower_bound_loss[0, :], label="Lower Bound Loss")
    ax.semilogx(distance_2D[0, :], median_bound_loss[0, :], label="Median Bound Loss")
    ax.semilogx(distance_2D[0, :], upper_bound_loss[0, :], label="Upper Bound Loss")

    plt.title("SHF - Super High Frequency Path Loss")
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.xlim((0, distance_2D[0, -1]))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid()

    plt.show()

