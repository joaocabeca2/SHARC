# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:45:50 2017

@author: edgar
"""
from multipledispatch import dispatch
import sys
import numpy as np

from sharc.station_manager import StationManager
from sharc.parameters.parameters import Parameters
from sharc.parameters.imt.parameters_indoor import ParametersIndoor
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_inh_office import PropagationInhOffice
from sharc.propagation.propagation_building_entry_loss import PropagationBuildingEntryLoss


class PropagationIndoor(Propagation):
    """
    This is a wrapper class which can be used for indoor simulations. It
    calculates the basic path loss between BS's and UE's of the same building,
    assuming 3 BS's per building. It also includes an additional building
    entry loss for the outdoor UE's that are served by indoor BS's.
    """

    # For BS and UE that are not in the same building, this value is assigned
    # so this kind of inter-building interference will not be effectivelly
    # taken into account during SINR calculations. This is assumption
    # simplifies the implementation and it is reasonable: intra-building
    # interference is much higher than inter-building interference
    HIGH_PATH_LOSS = 400

    def __init__(self, random_number_gen: np.random.RandomState, param: ParametersIndoor, ue_per_cell):
        super().__init__(random_number_gen)

        if param.basic_path_loss == "FSPL":
            self.bpl = PropagationFreeSpace(random_number_gen)
        elif param.basic_path_loss == "INH_OFFICE":
            self.bpl = PropagationInhOffice(random_number_gen)
        else:
            sys.stderr.write(
                "ERROR\nInvalid indoor basic path loss model: " + param.basic_path_loss,
            )
            sys.exit(1)

        self.bel = PropagationBuildingEntryLoss(random_number_gen)
        self.building_class = param.building_class
        self.bs_per_building = param.num_cells
        self.ue_per_building = ue_per_cell * param.num_cells

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
        """Wrapper function for the get_loss method to fit the Propagation ABC class interface
        Calculates the loss between station_a and station_b

        Parameters
        ----------
        params : Parameters
            Simulation parameters needed for the propagation class
        frequency: float
            Center frequency
        station_a : StationManager
            StationManager container
        station_b : StationManager
            StationManager container
        station_a_gains: np.ndarray defaults to None
            Not used
        station_b_gains: np.ndarray defaults to None
            Not used

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

        if wrap_around_enabled:
            bs_to_ue_dist_2d, bs_to_ue_dist_3d, _, _ = \
                station_b.get_dist_angles_wrap_around(station_a)
        else:
            bs_to_ue_dist_2d = station_b.get_distance_to(station_a)
            bs_to_ue_dist_3d = station_b.get_3d_distance_to(station_a)

        frequency_array = frequency * np.ones(bs_to_ue_dist_2d.shape)
        indoor_stations = np.tile(
            station_a.indoor, (station_b.num_stations, 1),
        )
        elevation = np.transpose(station_a.get_elevation(station_b))

        return self.get_loss(
            bs_to_ue_dist_3d,
            bs_to_ue_dist_2d,
            frequency_array,
            elevation,
            indoor_stations,
            params.imt.shadowing,
        )

    # pylint: disable=function-redefined
    # pylint: disable=arguments-renamed
    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool)
    def get_loss(
        self, distance_3D: np.ndarray, distance_2D: np.ndarray, frequency: float,
        elevation: np.ndarray, indoor_stations: np.ndarray, shadowing_flag: bool,
    ) -> np.array:
        """
        Calculates path loss for LOS and NLOS cases with respective shadowing
        (if shadowing has to be added)

        Parameters
        ----------
            distance_3D (np.array) : 3D distances between stations
            distance_2D (np.array) : 2D distances between stations
            elevation (np.array) : elevation angles from UE's to BS's
            frequency (np.array) : center frequencie [MHz]
            indoor (np.array) : indicates whether UE is indoor
            shadowing (bool) : if shadowing should be added or not

        Returns
        -------
            array with path loss values with dimensions of distance_2D

        """
        loss = PropagationIndoor.HIGH_PATH_LOSS * np.ones(frequency.shape)
        iter = int(frequency.shape[0] / self.bs_per_building)
        for i in range(iter):
            bi = int(self.bs_per_building * i)
            bf = int(self.bs_per_building * (i + 1))
            ui = int(self.ue_per_building * i)
            uf = int(self.ue_per_building * (i + 1))

            # calculate basic path loss
            loss[bi:bf, ui:uf] = self.bpl.get_loss(
                distance_3D=distance_3D[bi:bf, ui:uf],
                distance_2D=distance_2D[bi:bf, ui:uf],
                frequency=frequency[bi:bf, ui:uf],
                indoor=indoor_stations[0, ui:uf],
                shadowing=shadowing_flag,
            )

            # calculates the additional building entry loss for outdoor UE's
            # that are served by indoor BS's
            bel = (~ indoor_stations[0, ui:uf]) * self.bel.get_loss(
                frequency[bi:bf, ui:uf], elevation[bi:bf, ui:uf], "RANDOM", self.building_class,
            )

            loss[bi:bf, ui:uf] = loss[bi:bf, ui:uf] + bel

        return loss


if __name__ == '__main__':
    params = ParametersIndoor()
    params.basic_path_loss = "INH_OFFICE"
    params.n_rows = 3
    params.n_colums = 1
#    params.street_width = 30
    params.ue_indoor_percent = .95
    params.num_cells = 3
    params.building_class = "TRADITIONAL"

    bs_per_building = 3
    ue_per_bs = 3

    num_bs = bs_per_building * params.n_rows * params.n_colums
    num_ue = num_bs * ue_per_bs
    distance_2D = 150 * np.random.random((num_bs, num_ue))
    frequency = 27000 * np.ones(distance_2D.shape)
    indoor = np.random.rand(1, num_ue) < params.ue_indoor_percent
    indoor = np.tile(indoor, (num_bs, 1))
    h_bs = 3 * np.ones(num_bs)
    h_ue = 1.5 * np.ones(num_ue)
    distance_3D = np.sqrt(distance_2D**2 + (h_bs[:, np.newaxis] - h_ue)**2)
    height_diff = np.tile(h_bs, (num_bs, 3)) - np.tile(h_ue, (num_bs, 1))
    elevation = np.degrees(np.arctan(height_diff / distance_2D))

    propagation_indoor = PropagationIndoor(
        np.random.RandomState(), params, ue_per_bs,
    )
    loss_indoor = propagation_indoor.get_loss(
        distance_3D,
        distance_2D,
        frequency,
        elevation,
        indoor,
        False,
    )
    print(loss_indoor)
