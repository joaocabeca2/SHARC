import numpy as np

from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
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

        free_space_prop = PropagationFreeSpace(self.random_number_gen)
        free_space_loss = free_space_prop.get_free_space_loss(frequency * 1000, distances_3d)
        frequency *= np.ones(distances_2d.shape)
        ref_dist = 1.0 * np.ones(distances_2d.shape)
        n = 2.0 * np.ones(distances_2d.shape)
        
        gas_att = self.get_attenuation_geseous(gamma_o, gamma_w)
        rain_att = self.get_rain_attenuation(gamma_o, gamma_w)

        #With directional antennas, the basic transmission loss when the boresights of the antennas are aligned
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

    def get_rain_attenuation(self, gamma_o, gamma_w) -> np.array:
        """
        Calculates the total specific attenuation using a simplified linear formula.

        Parameters:
        - gamma_o (float): Specific attenuation due to dry air (in dB/km). 
        - gamma_w (float): Specific attenuation due to water vapor (in dB/km). 

        Returns:
        - float: The total specific attenuation (in dB/km).
        """
    def get_gaseous_attenuation(self, gamma_o, gamma_w) -> np.array:
        """
        Calculates the total specific attenuation using a simplified linear formula.

        Parameters:
        - gamma_o (float): Specific attenuation due to dry air (in dB/km). 
        - gamma_w (float): Specific attenuation due to water vapor (in dB/km). 

        Returns:
        - float: The total specific attenuation (in dB/km).
        """


