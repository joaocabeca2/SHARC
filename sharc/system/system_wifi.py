import os
import sys

import numpy as np

from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.topology.topology_hotspot import TopologyHotspot


class SystemWifi():
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem) -> None:
        self.parameters = param
        self.topology = self.generate_topology()
        self.topology.calculate_coordinates()

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

    def generate_aps(self, random_number_gen: np.random.RandomState) -> StationManager:
        param_ant = self.parameters.ap.antenna
        num_aps = self.topology.num_base_stations
        access_points = StationManager(num_aps)
        access_points.station_type = StationType.WIFI_APS

        access_points.x = self.topology.x
        access_points.y = self.topology.y
        access_points.elevation = -param_ant.downtilt * np.ones(num_aps)
        access_points.height = self.parameters.ap.height
        
        access_points.azimuth = self.topology.azimuth

        # Inicializa todos os APs como inativos
        access_points.active = np.zeros(num_aps, dtype=bool)

        access_points.tx_power = self.parameters.ap.conducted_power * np.ones(num_aps)
        access_points.rx_power = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.rx_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.ext_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.total_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )

        access_points.snr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.sinr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.sinr_ext = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        access_points.inr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )

        access_points.antenna = np.empty(
            num_aps, dtype=AntennaBeamformingImt,
        )

        for i in range(num_aps):
            access_points.antenna[i] = \
                AntennaBeamformingImt(
                    param_ant, access_points.azimuth[i],
                    access_points.elevation[i],
                )

        # access_points.antenna = [AntennaOmni(0) for ap in range(num_aps)]
        access_points.bandwidth = self.parameters.bandwidth * np.ones(num_aps)
        access_points.center_freq = self.parameters.frequency * np.ones(num_aps)
        access_points.noise_figure = self.parameters.ap.noise_figure * \
            np.ones(num_aps)
        access_points.thermal_noise = -500 * np.ones(num_aps)

        access_points.intersite_dist = self.parameters.topology.hotspot.intersite_distance
            
        return access_points

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
            raise ValstaError("Invalid type or length for parameter azimuth_range")

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
                deterministic_cell = Trsta

                if self.parameters.sta.distribution_type.upper() == "CELL":
                    central_cell = Trsta


            [sta_x, sta_y, theta, distance] = self.get_random_position(
                num_sta, self.topology, random_number_gen,
                self.parameters.minimum_separation_distance_ap_sta,
                deterministic_cell=Trsta,
            )
            psi = np.degrees(np.arctan((self.parameters.ap.height - self.parameters.sta.height) / distance))

            sta.azimuth = (azimuth + theta + np.pi / 2)
            sta.elevation = elevation + psi

        elif self.parameters.sta.distribution_type.upper() == "ANGLE_AND_DISTANCE":
            # Handle Rayleigh or Uniform distribution for STA distance
            if self.parameters.sta.distribution_distance.upper() == "RAYLEIGH":
                radius_scale = self.topology.cell_radius / 3.0345
                radius = random_number_gen.rayleigh(radius_scale, num_sta)
            elif self.parameters.sta.distribution_distance.upper() == "UNIFORM":
                radius = self.topology.cell_radius * random_number_gen.random_sample(num_sta)
            else:
                sys.stderr.write("ERROR\nInvalid STA distance distribution: " + self.parameters.sta.distribution_distance)
                sys.exit(1)

            if self.parameters.sta.distribution_azimuth.upper() == "NORMAL":
                N = 1.4
                angle_scale = 30
                angle_mean = 0
                angle_n = random_number_gen.normal(angle_mean, angle_scale, int(N * num_sta))

                angle_cutoff = np.max(azimuth_range)
                idx = np.where((angle_n < angle_cutoff) & (angle_n > -angle_cutoff))[0][:num_sta]
                angle = angle_n[idx]
            elif self.parameters.sta.distribution_azimuth.upper() == "UNIFORM":
                azimuth_range = (-60, 60)
                angle = (azimuth_range[1] - azimuth_range[0]) * random_number_gen.random_sample(num_sta) + azimuth_range[0]
            else:
                sys.stderr.write("ERROR\nInvalid STA azimuth distribution: " + self.parameters.sta.distribution_distance)
                sys.exit(1)

            # Calculate STA positions and azimuth/elevation for outdoor environment
            for ap in range(num_ap):
                idx = [i for i in range(
                    ap * num_sta_per_ap, ap * num_sta_per_ap + num_sta_per_ap,
                )]
                theta = self.topology.azimuth[ap] + angle[idx]
                x = self.topology.x[ap] + radius[idx] * np.cos(np.radians(theta))
                y = self.topology.y[ap] + radius[idx] * np.sin(np.radians(theta))
                sta_x.extend(x)
                sta_y.extend(y)

                sta.azimuth[idx] = (azimuth[idx] + theta + 180) % 360
                distance = np.sqrt((self.topology.x[ap] - x) ** 2 + (self.topology.y[ap] - y) ** 2)
                psi = np.degrees(np.arctan((self.parameters.ap.height - self.parameters.sta.height) / distance))
                sta.elevation[idx] = elevation[idx] + psi
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

        return sta
    
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
            Number of UE stations
        topology : Topology
            The IMT topology object
        random_number_gen : np.random.RandomState
            Random number generator
        min_dist_to_ap : _type_, optional
            Minimum distance to the BS, by default 0.
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

    def connect_stations(self):
        # Create a boolean list that list which links between tx and rx are active
        pass

    def calculate_coupling_loss(self):
        # calculate the coupling loss between stations on active links
        pass

    def calculate_gains(self):
        # calculate the gains between stations on active links
        pass


'''# Logicamente, eles são ativados após verificar o canal
        for ap in range(num_aps):
            if channel_is_free(ap):  # Simulação de verificação do canal
                access_points.active[ap] = Trsta'''