import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))

import numpy as np
from sharc.parameters.parameters import Parameters
from sharc.parameters.parameters_p1411 import ParametersP1411
from sharc.parameters.parameters_p452 import ParametersP452
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_p1411_12 import PropagationP1411
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_clear_air_452 import PropagationClearAir
from sharc.station_manager import StationManager


class PropagationEHF(Propagation):
    """
    A class for calculating millimetre-wave propagation losses at frequencies above 10 GHz,
    especially in the Extremely High Frequency (EHF) band described in ITU-R P1411-2
    """
    def __init__(
        self,
        random_number_gen: np.random.RandomState,
        los_adjustment_factor: float,
        model_params: ParametersP1411
    ):
        super().__init__(random_number_gen)
        self.los_adjustment_factor = los_adjustment_factor
        self.model_params = model_params

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
        
        distance = station_a.get_3d_distance_to(
            station_b,
        ) * (1e-3)  # P.452 expects Kms
        frequency_array = frequency * \
            np.ones(distance.shape) * (1e-3)  # P.452 expects GHz
        indoor_stations = np.tile(
            station_b.indoor, (station_a.num_stations, 1),
        )
        elevation = station_b.get_elevation(station_a)
        if params.imt.interfered_with:
            tx_gain = station_a_gains
            rx_gain = station_b_gains
        else:
            tx_gain = station_b_gains
            rx_gain = station_a_gains
        
        los_probability = self.get_los_probability(
            distance_2D, self.los_adjustment_factor,
        )
        los_condition = self.get_los_condition(los_probability)

        i_los = np.where(los_condition == True)[:2]
        i_nlos = np.where(los_condition == False)[:2]

        free_space_loss = self.get_free_space_loss(frequency, distances_3d, elevation, tx_gain, rx_gain)
        frequency *= np.ones(distances_2d.shape)
        ref_dist = 1.0 * np.ones(distances_2d.shape)
        n = 2.0 * np.ones(distances_2d.shape)
        
        gas_att = self.get_attenuation_geseous(distance, frequency_array, indoor_stations)
        rain_att = self.get_rain_attenuation() * np.ones(distances_2d.shape)

        loss = np.empty(distance_2D.shape)

        if len(i_los[0]):
            loss_los = self.get_loss_los(
                distance_2D, distance_3D, frequency, bs_height, ue_height, h_e, shadowing_los,
            )
            loss[i_los] = loss_los[i_los]

        if len(i_nlos[0]):
            loss_nlos = self.get_loss_nlos(
                distance_2D, distance_3D, frequency, bs_height, ue_height, h_e, shadowing_nlos,
            )
            loss[i_nlos] = loss_nlos[i_nlos]

        return loss
    
    def get_loss_los(self,
                frequency:np.array,
                distance_3d:np.array,
                reference_distance:np.array,
                free_space_loss:np.array,
                gas_attenuation:np.array,
                rain_attenuation:np.array,
                n:np.array):
        """
        Calculate the line-of-sight (LoS) transmission loss in decibels (dB)
        for millimeter-wave propagation with directional antennas.

        Parameters:
        -----------
        frequency : np.array
            The frequency of the transmitted signal in MHz.
        reference_distance: np.array
            reference distance of 1 meter
        distance_3d : np,array
            The distance between Station 1 and Station 2 in meters.
        gas_attenuation : np.array
            The attenuation caused by atmospheric gases in dB.
        rain_attenuation : np.array
            The attenuation caused by rain in dB.
        n : np.array
            The basic transmission loss exponent (default is 2 for free-space propagation).
        """

        return (
            free_space_loss + (10 * n * np.log10((distance_3d / reference_distance)
            + gas_attenuation + rain_attenuation))
        )
    
    def get_loss_nlos(self):
        """
        Calculate the line-of-sight (LoS) transmission loss in decibels (dB)
        for millimeter-wave propagation with directional antennas.
        """

        wavelength = self.model_params.wavelength
        d_corner = self.model_params.d_corner
        street_width1 = self.model_params.street_width1
        distance1 = self.model_params.distance1
        distance2 = self.model_params.distance2

        l_corner = self.calculate_Lcorner(d_corner, street_width1, distance2)
        l_att = self.calculate_Latt(d_corner, street_width1, distance2, distance1)

        return 20 * np.log10(wavelength / (2 * np.pi * 20)) + l_corner + l_att


    def calculate_Lcorner(self, d_corner, street_width1, distance2):
        """
        Calculates the corner loss based on the environment.

        Returns:
        - float: Corner loss in dB.
        """

        L_c = 20 if self.environment == "urban" else 30  # Base corner loss depending on the environment

        if distance2 > (street_width1 /(2 + 1 + d_corner)):
        # Beyond the corner region, use the fixed L_c value
            return L_c
        
        elif distance2 >= (street_width1 / 3) and distance2 <= (street_width1 / (2 + 1 + d_corner)):
            # Within the corner region, calculate dynamically
            return (L_c / np.log10(1 + d_corner)) * np.log10(distance2 - street_width1 / 2)
        
        else:
            # If x2 is out of the expected range
            raise ValueError("x2 must be greater than or equal to w1/2 + 1.")
        
    def calculate_Latt(self, d_corner, street_width1, distance2, distance1):
        """
        Calculates the attenuation in the NLoS region beyond the corner.

        Returns:
        - float: NLoS attenuation (in dB).
        """

        if distance2 <= (street_width1 / 2 + 1 + d_corner):
            return 0
        else:
            beta = 6  # Coefficient for urban and residential environments
            return 10 * beta * np.log10((distance1 + distance2) / ((distance1 + street_width1) /(2 + d_corner)))
    
    def get_los_condition(self, p_los: np.array) -> np.array:
        """
        Evaluates if user equipments are LOS (True) of NLOS (False).

        Parameters
        ----------
            p_los : array with LOS probabilities for each user equipment.

        Returns
        -------
            An array with True or False if user equipments are in LOS of NLOS
            condition, respectively.
        """
        los_condition = self.random_number_gen.random_sample(
            p_los.shape,
        ) < p_los
        return los_condition
    
    def get_los_probability(
        self,
        distance_2D: np.array,
        los_adjustment_factor: float,
    ) -> np.array:
        """
        Returns the line-of-sight (LOS) probability

        Parameters
        ----------
            distance_2D : Two-dimensional array with 2D distance values from
                          base station to user terminal [m]
            los_adjustment_factor : adjustment factor to increase/decrease the
                          LOS probability. Original value is 18 as per 3GPP

        Returns
        -------
            LOS probability as a numpy array with same length as distance
        """

        p_los = np.ones(distance_2D.shape)
        idl = np.where(distance_2D > los_adjustment_factor)
        p_los[idl] = (
            los_adjustment_factor / distance_2D[idl] +
            np.exp(-distance_2D[idl] / 36) * (1 - los_adjustment_factor / distance_2D[idl])
        )

        return p_los

    def get_rain_attenuation(self) -> np.array:
        return 0.0

    def get_gaseous_attenuation(self, distance: np.array,
                                frequency: np.array,
                                indoor_stations: np.array,
                                elevation: np.array,
                                tx_gain: np.array,
                                rx_gain: np.array) -> np.array:
        gaseous_att = PropagationClearAir(self.random_number_gen, ParametersP452)
        return gaseous_att.get_loss(distance, frequency, indoor_stations, elevation, tx_gain, rx_gain)
    
    def get_free_space_loss(self, frequency: np.array, distance_3d: np.array) -> np.array:
        free_space_prop = PropagationFreeSpace(self.random_number_gen)
        return free_space_prop.get_free_space_loss(frequency * 1000, distance_3d)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Configuração de parâmetros
    num_ue = 1
    num_bs = 1000
    h_bs = 6 * np.ones(num_bs)
    h_ue = 1.5 * np.ones(num_ue)
    

    # Configuração da distância para o cenário
    distance_2D = np.repeat(np.linspace(5, 660, num=num_bs)[np.newaxis, :], num_ue, axis=0)
    frequency = 7 * np.ones(num_bs)  
    distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[np.newaxis, :])**2)

    # Gerador de números aleatórios
    random_number_gen = np.random.RandomState(101)
    ref_dist = np.asarray(1.0 * np.ones(distance_2D.shape))
    n = np.asarray(2 * np.ones(distance_2D.shape))

    indoor_stations = np.asarray(np.zeros(distance_3D.shape, dtype=bool))
    tx_gain = 0 * np.ones(distance_2D.shape)
    rx_gain = 0 * np.ones(distance_2D.shape)
    
    elevation = np.empty(num_bs)
    
    ehf = PropagationEHF(random_number_gen)
    free_space_prop = PropagationFreeSpace(random_number_gen)
    free_space_loss = free_space_prop.get_loss(distance_3D, frequency * 1000)
    gas_att = ehf.get_gaseous_attenuation(np.asarray(distance_3D), np.asarray(frequency), indoor_stations, elevation, tx_gain, rx_gain)
    rain_att = 0 * np.ones(distance_2D.shape)

    p1411 = PropagationP1411(random_number_gen , 'Urban')
    p1411_loss = p1411.calculate_median_basic_loss(distance_3D, frequency, random_number_gen)

    loss = ehf.get_loss_los(frequency, distance_3D, ref_dist, free_space_loss, gas_att, rain_att, n)
    

    # Plotando os gráficos
    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
    ax = fig.gca()
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b']))

    ax.semilogx(distance_2D[0, :], loss[0, :], label="Milimetric wave loss (LOS)")
    ax.semilogx(distance_2D[0, :], p1411_loss[0, :], label="Median basic loss")
    ax.semilogx(distance_2D[0, :], free_space_loss[0, :], label="Free space loss")

    plt.title('Milimetric wave loss')
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.xlim((0, distance_2D[0, -1]))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid()

    plt.show()