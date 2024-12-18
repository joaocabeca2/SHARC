import math
import os
import sys

import numpy as np

from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.propagation.propagation_factory import PropagationFactory
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.topology.topology_hotspot import TopologyHotspot


class SystemWifi():
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem) -> None:
        self.parameters = param
        self.topology = self.generate_topology()
        self.topology.calculate_coordinates()
        num_aps = self.topology.num_base_stations
        num_stas = num_aps * self.parameters.sta.k * self.parameters.sta.k_m

        self.ap_power_gain = 10 * math.log10(
            self.parameters.ap.antenna.n_rows *
            self.parameters.ap.antenna.n_columns,
        )
        self.sta_power_gain = 10 * math.log10(
            self.parameters.sta.antenna.n_rows *
            self.parameters.sta.antenna.n_columns,
        )
        self.ap_antenna_gain = list()
        self.sta_antenna_gain = list()
        self.path_loss = np.empty([num_aps, num_stas])
        self.coupling_loss = np.empty([num_aps, num_stas])

        self.ap_to_sta_phi = np.empty([num_aps, num_stas])
        self.ap_to_sta_theta = np.empty([num_aps, num_stas])
        self.ap_to_sta_beam_rbs = -1.0 * np.ones(num_stas, dtype=int)

        self.sta = np.empty(num_stas)
        self.ap = np.empty(num_aps)

        self.link = dict([(bs, list()) for bs in range(num_aps)])

        self.num_rb_per_bs = math.trunc(
            (1 - self.parameters.guard_band_ratio) *
            self.parameters.bandwidth / self.parameters.rb_bandwidth,
        )
        # calculates the number of RB per STA on a given AP
        self.num_rb_per_sta = math.trunc(
            self.num_rb_per_bs / self.parameters.sta.k,
        )


        if hasattr(self.parameters, "polarization_loss"):
            self.polarization_loss = self.param_system.polarization_loss
        else:
            self.polarization_loss = 3.0


    def generate_topology(self):
        if self.parameters.topology.type == "HOTSPOT":
            return TopologyHotspot(
                self.parameters.topology.hotspot,
                self.parameters.topology.hotspot.intersite_distance,
                self.parameters.topology.hotspot.num_clusters
            )
        sys.stderr.write(
            "ERROR\nInvalid topology: " +
            self.parameters.topology.type,
        )
        sys.exit(1)
    
    def generate_propagation(self):
        return PropagationFactory.create_propagation(
            self.parameters.channel_model,
            self.parameters,
            self.parameters
        )
    def generate_aps(self, random_number_gen: np.random.RandomState) -> StationManager:
        param_ant = self.parameters.ap.antenna
        num_aps = self.topology.num_base_stations
        access_points = StationManager(num_aps)
        access_points.station_type = StationType.WIFI_APS

        # Gerar posições aleatórias para os APs
        [access_points_x, access_points_y, theta, distance] = self.get_random_position(
            num_aps, self.topology, random_number_gen,
            self.parameters.minimum_separation_distance_ap_sta,  # Distância mínima
            deterministic_cell=False  # Permitir células aleatórias
        )

        access_points.x = np.array(access_points_x)
        access_points.y = np.array(access_points_y)
        access_points.elevation = -param_ant.downtilt * np.ones(num_aps)
        access_points.height = self.parameters.ap.height * np.ones(num_aps)
        access_points.azimuth = theta  # Usar o ângulo calculado aleatoriamente

        access_points.active = random_number_gen.rand(num_aps) < self.parameters.ap.load_probability
        access_points.tx_power = self.parameters.ap.conducted_power * np.ones(num_aps)
        access_points.rx_power = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.rx_interference = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.ext_interference = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.total_interference = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.snr = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.sinr = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.sinr_ext = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])
        access_points.inr = dict([(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)])


        for i in range(num_aps):
            access_points.antenna[i] = \
                AntennaBeamformingImt(
                    param_ant, access_points.azimuth[i],
                    access_points.elevation[i],
                )
        
        access_points.bandwidth = self.parameters.bandwidth * np.ones(num_aps)
        access_points.center_freq = self.parameters.frequency * np.ones(num_aps)
        access_points.noise_figure = self.parameters.ap.noise_figure * np.ones(num_aps)
        access_points.thermal_noise = -500 * np.ones(num_aps)
        access_points.intersite_dist = self.parameters.topology.hotspot.intersite_distance
            
        self.ap = access_points

    def generate_stas(self, random_number_gen: np.random.RandomState) -> StationManager:
        num_ap = self.topology.num_base_stations
        num_sta_per_ap = self.parameters.sta.k * self.parameters.sta.k_m
        num_sta = num_ap * num_sta_per_ap

        sta = StationManager(num_sta)
        sta.station_type = StationType.WIFI_STA

        sta_x = list()
        sta_y = list()

        # TODO: Sanitaze the azimuth_range parameter
        azimuth_range = self.parameters.sta.azimuth_range
        if (not isinstance(azimuth_range, tuple)) or len(azimuth_range) != 2:
            raise ValueError("Invalid type or length for parameter azimuth_range")

        # Calculate STA pointing (azimuth and elevation)
        azimuth = (azimuth_range[1] - azimuth_range[0]) * random_number_gen.random_sample(num_sta) + azimuth_range[0]
        elevation_range = (-90, 90)
        elevation = (elevation_range[1] - elevation_range[0]) * random_number_gen.random_sample(num_sta) + elevation_range[0]

        if self.parameters.sta.distribution_type.upper() == "UNIFORM" or \
        self.parameters.sta.distribution_type.upper() == "CELL" or \
        self.parameters.sta.distribution_type.upper() == "UNIFORM_IN_CELL":

            deterministic_cell = False
            central_cell = False

            if self.parameters.sta.distribution_type.upper() == "UNIFORM_IN_CELL" or \
            self.parameters.sta.distribution_type.upper() == "CELL":
                deterministic_cell = True

                if self.parameters.sta.distribution_type.upper() == "CELL":
                    central_cell = True


            [sta_x, sta_y, theta, distance] = self.get_random_position(
                num_sta, self.topology, random_number_gen,
                self.parameters.minimum_separation_distance_ap_sta,
                deterministic_cell=True,
            )
            psi = np.degrees(np.arctan((self.parameters.ap.height - self.parameters.sta.height) / distance))

            sta.azimuth = (azimuth + theta + np.pi / 2)
            sta.elevation = elevation + psi


        else:
            sys.stderr.write("ERROR\nInvalid STA distribution type: " + self.parameters.sta.distribution_type)
            sys.exit(1)

        sta.x = np.array(sta_x)
        sta.y = np.array(sta_y)

        sta.active = np.zeros(num_sta, dtype=bool)
        sta.height = self.parameters.sta.height * np.ones(num_sta)
        sta.indoor = random_number_gen.random_sample(num_sta) <= (self.parameters.sta.indoor_percent / 100)
        sta.rx_interference = -500 * np.ones(num_sta)
        sta.ext_interference = -500 * np.ones(num_sta)

        # Antenna parameters for the STAs
        par = self.parameters.sta.antenna.get_antenna_parameters()
        for i in range(num_sta):
            sta.antenna[i] = AntennaBeamformingImt(par, sta.azimuth[i], sta.elevation[i])

        sta.bandwidth = self.parameters.bandwidth * np.ones(num_sta)
        sta.center_freq = self.parameters.frequency * np.ones(num_sta)
        sta.noise_figure = self.parameters.sta.noise_figure * np.ones(num_sta)

        sta.intersite_dist = self.parameters.topology.hotspot.intersite_distance

        self.sta = sta
    
    def get_random_position(self, num_sta: int,
                            topology: TopologyHotspot,
                            random_number_gen: np.random.RandomState,
                            min_dist_to_ap=0.,
                            central_cell=False,
                            deterministic_cell=False):
        """
        Generate sta random-possitions inside the topolgy area.

        Parameters
        ----------
        num_sta : int
            Number of stations
        topology : Topology
            The wifi topology object
        random_number_gen : np.random.RandomState
            Random number generator
        min_dist_to_ap : _type_, optional
            Minimum distance to the ap, by default 0.
        central_cell : bool, optional
            Whether the central cell in the cluster is used, by default False
        deterministic_cell : bool, optional
            Fix the cell to be used as anchor point, by default False

        Returns
        -------
        tuple
            x, y, azimuth and elevation angles.
        """
        hexagon_radius = self.topology.intersite_distance / 3

        x = np.array([])
        y = np.array([])
        ap_x = -hexagon_radius
        ap_y = 0

        while len(x) < num_sta:
            num_sta_temp = num_sta - len(x)
            # generate UE uniformly in a triangle
            x_temp = random_number_gen.uniform(0, hexagon_radius * np.cos(np.pi / 6), num_sta_temp)
            y_temp = random_number_gen.uniform(0, hexagon_radius / 2, num_sta_temp)

            invert_index = np.arctan(y_temp / x_temp) > np.pi / 6
            y_temp[invert_index] = -(hexagon_radius / 2 - y_temp[invert_index])
            x_temp[invert_index] = (hexagon_radius * np.cos(np.pi / 6) - x_temp[invert_index])

            # randomly choose a hextant
            hextant = random_number_gen.random_integers(0, 5, num_sta_temp)
            hextant_angle = np.pi / 6 + np.pi / 3 * hextant

            old_x = x_temp
            x_temp = x_temp * np.cos(hextant_angle) - y_temp * np.sin(hextant_angle)
            y_temp = old_x * np.sin(hextant_angle) + y_temp * np.cos(hextant_angle)

            dist = np.sqrt((x_temp - ap_x) ** 2 + (y_temp - ap_y) ** 2)
            indices = dist > min_dist_to_ap

            x_temp = x_temp[indices]
            y_temp = y_temp[indices]

            x = np.append(x, x_temp)
            y = np.append(y, y_temp)

        x = x - ap_x
        y = y - ap_y

        # choose cells
        if central_cell:
            central_cell_indices = np.where((topology.x == 0) & (topology.y == 0))

            if not len(central_cell_indices[0]):
                sys.stderr.write("ERROR\nTopology does not have a central cell")
                sys.exit(1)

            cell = central_cell_indices[0][random_number_gen.random_integers(0, len(central_cell_indices[0]) - 1,
                                                                             num_sta)]
        elif deterministic_cell:
            num_ap = topology.num_base_stations
            sta_per_cell = num_sta / num_ap
            cell = np.repeat(np.arange(num_ap, dtype=int), sta_per_cell)

        else:  # random cells
            num_ap = topology.num_base_stations
            cell = random_number_gen.random_integers(0, num_ap - 1, num_sta)

        cell_x = topology.x[cell]
        cell_y = topology.y[cell]

        azimuth_rad = topology.azimuth * np.pi / 180

        # rotqte
        x_old = x
        x = cell_x + x * np.cos(azimuth_rad[cell]) - y * np.sin(azimuth_rad[cell])
        y = cell_y + x_old * np.sin(azimuth_rad[cell]) + y * np.cos(azimuth_rad[cell])

        x = list(x)
        y = list(y)

       
        theta = np.arctan2(y - cell_y, x - cell_x)

        # calculate elevation angle
        # psi is the vertical angle of the UE wrt the serving BS
        distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)

        return x, y, theta, distance

    def connect_aps_to_stas(self):
        num_stas_per_ap = self.parameters.sta.k * self.parameters.sta.k_m
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            sta_list = [
                i for i in range(
                    ap * num_stas_per_ap, ap * num_stas_per_ap + num_stas_per_ap,
                )
            ]
            self.link[ap] = sta_list

    def select_sta(self, random_number_gen: np.random.RandomState):
        """
        Select K STAs randomly from all the STAs linked to one AP as “chosen”
        STAs. These K “chosen” STAs will be scheduled during this snapshot.
        """

        self.ap_to_sta_d_2D = self.ap.get_distance_to(self.sta)
        self.ap_to_sta_d_3D = self.ap.get_3d_distance_to(self.sta)
        self.ap_to_sta_phi, self.ap_to_sta_theta = self.ap.get_pointing_vector_to(
            self.sta,
        )

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            # select K STAs among the ones that are connected to AP
            random_number_gen.shuffle(self.link[ap])
            K = self.parameters.wifi.sta.k
            del self.link[ap][K:]
            # Activate the selected STAs and create beams
            if self.ap.active[ap]:
                self.sta.active[self.link[ap]] = np.ones(K, dtype=bool)
                for sta in self.link[ap]:
                    # add beam to AP antennas
                    self.ap.antenna[ap].add_beam(
                        self.ap_to_sta_phi[ap, sta],
                        self.ap_to_sta_theta[ap, sta],
                    )
                    # add beam to STA antennas
                    self.sta.antenna[sta].add_beam(
                        self.ap_to_sta_phi[ap, sta] - 180,
                        180 - self.ap_to_sta_theta[ap, sta],
                    )
                    # set beam resource block group
                    self.ap_to_sta_beam_rbs[sta] = len(
                        self.ap.antenna[ap].beams
                    )

    def calculate_intra_wifi_coupling_loss(
        self,
    ) -> np.array:
        """
        Calculates the coupling loss (path loss + antenna gains + other losses) between
        Wi-Fi stations (STA and AP).

        Returns an numpy array with wifi_ap_station.size X wifi_sta_station.size with coupling loss
        values.

        Parameters
        ----------
        system_station : StationManager
            A StationManager object representing Wi-Fi STA stations
        wifi_station : StationManager
            A StationManager object representing Wi-Fi AP stations
        is_co_channel : bool, optional
            Whether the interference analysis is co-channel or not, by default True.
            This parameter is ignored. It's kept to maintain method interface.

        Returns
        -------
        np.array
            Returns an numpy array with wifi_ap_station.size X wifi_sta_station.size with coupling loss
            values.
        """
        # Calculate the antenna gains

        ant_gain_ap_to_sta = self.calculate_gains(
            self.ap, self.sta
        )
        ant_gain_sta_to_ap = self.calculate_gains(
            self.sta, self.ap
        )

        # Calculate the path loss between Wi-Fi stations. Primarily used for UL power control.

        # Note on the array dimensions for coupling loss calculations:
        # The function get_loss returns an array station_a x station_b
        path_loss = self.propagation_wifi.get_loss(
            self.parameters,
            self.parameters.wifi.frequency,
            self.sta,
            self.ap,
            ant_gain_sta_to_ap,
            ant_gain_ap_to_sta,
        )

        # Collect Wi-Fi AP and STA antenna gain samples
        self.path_loss_wifi = np.transpose(path_loss)
        self.wifi_ap_antenna_gain = ant_gain_ap_to_sta
        self.wifi_sta_antenna_gain = np.transpose(ant_gain_sta_to_ap)
        additional_loss = self.parameters.wifi.ap.ohmic_loss \
            + self.parameters.wifi.sta.ohmic_loss \
            + self.parameters.wifi.sta.body_loss

        # calculate coupling loss
        coupling_loss = np.squeeze(
            self.path_loss_wifi - self.wifi_ap_antenna_gain - self.wifi_sta_antenna_gain,
        ) + additional_loss

        return coupling_loss
    
    def calculate_coupling_loss(self):
        # calculate the coupling loss between stations on active links
        pass

    def calculate_gains(self):
       pass
