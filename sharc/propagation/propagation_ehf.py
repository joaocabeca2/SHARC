import numpy as np

from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
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
    ):
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

        free_space_loss = self.get_free_space_loss(frequency, distances_3d, elevation, tx_gain, rx_gain)
        frequency *= np.ones(distances_2d.shape)
        ref_dist = 1.0 * np.ones(distances_2d.shape)
        n = 2.0 * np.ones(distances_2d.shape)
        
        gas_att = self.get_attenuation_geseous(distance, frequency_array, indoor_stations)
        rain_att = self.get_rain_attenuation() * np.ones(distances_2d.shape)


        return self._get_loss(frequency, distances_3d, free_space_loss,
                            ref_dist, gas_att, rain_att, n)
    def _get_loss(self,
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
            free_space_loss + (10 * n * np.log10(distance_3d / reference_distance))
            + gas_attenuation + rain_attenuation
        )

    def get_rain_attenuation(self) -> np.array:
        return 0.0

    def get_gaseous_attenuation(self, distance: np.array,
                                frequency: np.array,
                                indoor_stations: np.array,
                                elevation: np.array,
                                tx_gain: np.array,
                                rx_gain: np.array) -> np.array:
        gaseous_att = PropagationClearAir()
        return gaseous_att.get_loss(distance, frequency, indoor_stations, elevation, tx_gain, rx_gain)
    
    def get_free_space_loss(self, frequency: np.array, distance_3d: np.array) -> np.array:
        free_space_prop = PropagationFreeSpace()
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

    ehf = PropagationEHF()

    free_space_loss = free_space_prop.get_free_space_loss(frequency * 1000, distance_3D)
    median_basic_loss = p1411.calculate_median_basic_loss(distance_3D, frequency, random_number_gen)
    excess_loss = p1411.calculate_excess_loss(free_space_loss, median_basic_loss, distance_3D)

    # Plotando os gráficos
    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
    ax = fig.gca()
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b']))

    ax.semilogx(distance_2D[0, :], median_basic_loss[0, :], label="Median basic Loss")
    ax.semilogx(distance_2D[0, :], excess_loss[0, :], label="Excess Loss")
    ax.semilogx(distance_2D[0, :], free_space_loss[0, :], label="Free space Loss")

    plt.title(p1411.environment)
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.xlim((0, distance_2D[0, -1]))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid()

    plt.show()