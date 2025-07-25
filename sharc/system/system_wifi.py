import math
import sys

import numpy as np
from scipy.stats.sampling import DiscreteAliasUrn
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.propagation.propagation_factory import PropagationFactory
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.topology.topology_factory import TopologyFactory
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.topology.topology_macrocell import TopologyMacrocell
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.topology.topology import Topology

class SystemWifi:
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem, param_ant_ap: ParametersAntennaWifi) -> None:
        self.parameters = param
        self.parameters_antenna = param_ant_ap
        self.topology = TopologyFactory.createTopology(self.parameters)
        self.topology.calculate_coordinates()
        self.num_aps = self.topology.num_base_stations
        self.num_stas = self.num_aps * self.parameters.sta.k * self.parameters.sta.k_m

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
        self.path_loss = np.empty([self.num_aps, self.num_stas])
        self.coupling_loss = np.empty([self.num_aps, self.num_stas])

        self.ap_to_sta_phi = np.empty([self.num_aps, self.num_stas])
        self.ap_to_sta_theta = np.empty([self.num_aps, self.num_stas])
        self.ap_to_sta_beam_rbs = -1.0 * np.ones(self.num_stas, dtype=int)

        self.sta = np.empty(self.num_stas)
        self.ap = np.empty(self.num_aps)

        self.link = dict([(bs, list()) for bs in range(self.num_aps)])

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

        self.propagation_wifi = self.generate_propagation()
        self.bandwidth = self.parameters.bandwidth
        self.noise_temperature = self.parameters.noise_temperature
    
    
    def generate_propagation(self):
        return PropagationFreeSpace(np.random.RandomState(1))
    
    def generate_aps(self, random_number_gen: np.random.RandomState) -> StationManager:
        param_ant = self.parameters_antenna.get_antenna_parameters()
        num_aps = self.topology.num_base_stations
        wifi_aps = StationManager(num_aps)
        wifi_aps.station_type = StationType.WIFI_APS

        wifi_aps.x = self.topology.x
        wifi_aps.y = self.topology.y
        wifi_aps.elevation = -param_ant.downtilt * np.ones(num_aps)
        wifi_aps.height = self.parameters.ap.height * np.ones(num_aps)

        wifi_aps.azimuth = self.topology.azimuth
        random_values = random_number_gen.rand(num_aps)
        wifi_aps.active = random_values < self.parameters.ap.load_probability
        wifi_aps.tx_power = self.parameters.ap.conducted_power * np.ones(num_aps)
        wifi_aps.rx_power = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.rx_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.ext_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.total_interference = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )

        wifi_aps.snr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.sinr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.sinr_ext = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )
        wifi_aps.inr = dict(
            [(ap, -500 * np.ones(self.parameters.sta.k)) for ap in range(num_aps)],
        )

        wifi_aps.antenna = np.empty(
            num_aps, dtype=AntennaOmni,
        )

        for i in range(num_aps):
            wifi_aps.antenna[i] = AntennaOmni()
    
        wifi_aps.bandwidth = self.parameters.bandwidth * np.ones(num_aps)
        wifi_aps.center_freq = self.parameters.frequency * np.ones(num_aps)
        wifi_aps.noise_figure = self.parameters.ap.noise_figure * np.ones(num_aps)
        wifi_aps.thermal_noise = -500 * np.ones(num_aps)


        if self.parameters.topology.type == 'HOTSPOT':
            wifi_aps.intersite_dist = self.parameters.topology.hotspot.intersite_distance

        self.wifi_ap = wifi_aps
        return wifi_aps

    def generate_stas(self, random_number_gen: np.random.RandomState) -> StationManager:
        num_sta_per_ap = self.parameters.sta.k * self.parameters.sta.k_m
        wifi_sta = StationManager(self.num_sta)
        wifi_sta.station_type = StationType.WIFI_STA

        sta_x = list()
        sta_y = list()
        sta_z = list()

        wifi_sta.indoor = np.ones(self.num_sta, dtype=bool)

        azimuth_range = (-60, 60)
        azimuth = (azimuth_range[1] - azimuth_range[0]) * \
            random_number_gen.random_sample(self.num_sta) + azimuth_range[0]

        elevation_range = (-90, 90)
        elevation = (elevation_range[1] - elevation_range[0]) * \
            random_number_gen.random_sample(self.num_sta) + elevation_range[0]

        delta_x = (
            self.topology.b_w / math.sqrt(self.topology.ue_indoor_percent) - self.topology.b_w
        ) / 2
        delta_y = (
            self.topology.b_d / math.sqrt(self.topology.ue_indoor_percent) - self.topology.b_d
        ) / 2

        for ap in range(self.num_aps):
            idx = [
                i for i in range(
                    ap * num_sta_per_ap, ap * num_sta_per_ap + num_sta_per_ap,
                )
            ]
            if ap % self.topology.num_cells == 0 and ap < self.topology.total_bs_level:
                x_min = self.topology.x[ap] - self.topology.cell_radius - delta_x
                x_max = self.topology.x[ap] + self.topology.cell_radius
            elif ap % self.topology.num_cells == self.topology.num_cells - 1 and ap < self.topology.total_bs_level:
                x_min = self.topology.x[ap] - self.topology.cell_radius
                x_max = self.topology.x[ap] + self.topology.cell_radius + delta_x
            else:
                x_min = self.topology.x[ap] - self.topology.cell_radius
                x_max = self.topology.x[ap] + self.topology.cell_radius

            if ap < self.topology.total_bs_level:
                y_min = self.topology.y[ap] - self.topology.b_d / 2 - delta_y
                y_max = self.topology.y[ap] + self.topology.b_d / 2 + delta_y
            else:
                y_min = self.topology.y[ap] - self.topology.b_d / 2
                y_max = self.topology.y[ap] + self.topology.b_d / 2

            x = (x_max - x_min) * \
                random_number_gen.random_sample(num_sta_per_ap) + x_min
            y = (y_max - y_min) * \
                random_number_gen.random_sample(num_sta_per_ap) + y_min
            z = [
                self.topology.height[ap] - self.topology.b_h +
                self.parameters.sta.height for k in range(num_sta_per_ap)
            ]
            sta_x.extend(x)
            sta_y.extend(y)
            sta_z.extend(z)

            theta = np.degrees(
                np.arctan2(
                    y - self.topology.y[ap], x - self.topology.x[ap],
                ),
            )
            wifi_sta.azimuth[idx] = (azimuth[idx] + theta + 180) % 360

            distance = np.sqrt(
                (self.topology.x[ap] - x)**2 + (self.topology.y[ap] - y)**2,
            )
            psi = np.degrees(
                np.arctan((self.parameters.sta.height - self.parameters.sta.height) / distance),
            )
            wifi_sta.elevation[idx] = elevation[idx] + psi

            if ap % self.topology.num_cells == 0:
                out = (x < self.topology.x[ap] - self.topology.cell_radius) | \
                    (y > self.topology.y[ap] + self.topology.b_d / 2) | \
                    (y < self.topology.y[ap] - self.topology.b_d / 2)
            elif ap % self.topology.num_cells == self.topology.num_cells - 1:
                out = (x > self.topology.x[ap] + self.topology.cell_radius) | \
                    (y > self.topology.y[ap] + self.topology.b_d / 2) | \
                    (y < self.topology.y[ap] - self.topology.b_d / 2)
            else:
                out = (y > self.topology.y[ap] + self.topology.b_d / 2) | \
                    (y < self.topology.y[ap] - self.topology.b_d / 2)
            self.wifi_sta.indoor[idx] = ~out

        wifi_sta.x = np.array(sta_x)
        wifi_sta.y = np.array(sta_y)
        wifi_sta.height = np.array(sta_z)

        wifi_sta.active = np.zeros(self.num_sta, dtype=bool)
        wifi_sta.rx_interference = -500 * np.ones(self.num_sta)
        wifi_sta.ext_interference = -500 * np.ones(self.num_sta)

        for i in range(self.num_sta):
            wifi_sta.antenna[i] = AntennaOmni()

        self.wifi_sta = wifi_sta
        return wifi_sta
    
    '''def get_random_position(self,
                            random_number_gen: np.random.RandomState,
                            min_dist_to_ap=0.,
                            central_cell=False,
                            deterministic_cell=False,
                            ):
        """
        Generate sta random-possitions inside the topolgy area.
        """
        hexagon_radius = topology.intersite_distance * 2 / 3

        x = np.array([])
        y = np.array([])
        z = np.array([])
        ap_x = -hexagon_radius
        ap_y = 0

        while len(x) < self.num_stas:
            num_stas_temp = self.num_stas - len(x)
            # generate UE uniformly in a triangle
            x_temp = random_number_gen.uniform(
                0, hexagon_radius * np.cos(np.pi / 6), num_stas_temp)
            y_temp = random_number_gen.uniform(
                0, hexagon_radius / 2, num_stas_temp)

            invert_index = np.arctan(y_temp / x_temp) > np.pi / 6
            y_temp[invert_index] = -(hexagon_radius / 2 - y_temp[invert_index])
            x_temp[invert_index] = (
                hexagon_radius *
                np.cos(
                    np.pi /
                    6) -
                x_temp[invert_index])

            # randomly choose a hextant
            hextant = random_number_gen.random_integers(0, 5, num_stas_temp)
            hextant_angle = np.pi / 6 + np.pi / 3 * hextant

            old_x = x_temp
            x_temp = x_temp * np.cos(hextant_angle) - \
                y_temp * np.sin(hextant_angle)
            y_temp = old_x * np.sin(hextant_angle) + \
                y_temp * np.cos(hextant_angle)

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
            central_cell_indices = np.where(
                (self.topology.x == 0) & (self.topology.y == 0))

            if not len(central_cell_indices[0]):
                sys.stderr.write(
                    "ERROR\nTopology does not have a central cell")
                sys.exit(1)

            cell = central_cell_indices[0][random_number_gen.random_integers(
                0, len(central_cell_indices[0]) - 1, self.num_stas)]
        elif deterministic_cell:
            stas_per_cell = self.num_stas / self.num_aps
            cell = np.repeat(np.arange(self.num_aps, dtype=int), stas_per_cell)

        else:  # random cells
            cell = random_number_gen.random_integers(0, self.num_aps - 1, self.num_stas)

        cell_x = self.topology.x[cell]
        cell_y = self.topology.y[cell]
        cell_z = self.topology.z[cell]

        # x = x + cell_x + hexagon_radius * np.cos(topology.azimuth[cell] * np.pi / 180)
        # y = y + cell_y + hexagon_radius * np.sin(topology.azimuth[cell] * np.pi / 180)
        old_x = x
        x = x * np.cos(np.radians(self.topology.azimuth[cell])) - \
            y * np.sin(np.radians(self.topology.azimuth[cell]))
        y = old_x * np.sin(np.radians(self.topology.azimuth[cell])) + y * np.cos(
            np.radians(self.topology.azimuth[cell]))
        x = x + cell_x
        y = y + cell_y
        z = cell_z

        x = list(x)
        y = list(y)
        z = list(z)

        # calculate UE azimuth wrt serving BS
        if self.topology.is_space_station is False:
            theta = np.arctan2(y - cell_y, x - cell_x)

            # calculate elevation angle
            # psi is the vertical angle of the UE wrt the serving BS
            distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)
        else:
            theta = np.arctan2(
                y - self.topology.space_station_y[cell],
                x - self.topology.space_station_x[cell])
            distance = np.sqrt((cell_x - x) ** 2 +
                               (cell_y - y) ** 2 + (cell_z)**2)

        return x, y, z, theta, distance'''

    def connect_wifi_sta_to_ap(self):
        """
        Link the Wi-Fi STA's to the serving AP. It is assumed that each group of K
        user equipments are distributed and pointed to a certain access point
        """
        num_sta_per_ap = self.parameters.wifi.sta.k * self.parameters.wifi.sta.k_m
        ap_active = np.where(self.wifi_ap.active)[0]
        for ap in ap_active:
            sta_list = [
                i for i in range(
                    ap * num_sta_per_ap, ap * num_sta_per_ap + num_sta_per_ap,
                )
            ]
            self.link[ap] = sta_list

    def select_sta(self, random_number_gen: np.random.RandomState):
        """
        Select K STAs randomly from all the STAs linked to one AP as “chosen”
        STAs. These K “chosen” STAs will be scheduled during this snapshot.
        """
        # Calculate distances and angles between Access Points (APs) and Stations (STAs)
        if self.wrap_around_enabled:
            self.ap_to_sta_d_2D, self.ap_to_sta_d_3D, self.ap_to_sta_phi, self.ap_to_sta_theta = \
                self.wifi_ap.get_dist_angles_wrap_around(self.sta)
        else:
            self.ap_to_sta_d_2D = self.wifi_ap.get_distance_to(self.wifi_sta)
            self.ap_to_sta_d_3D = self.wifi_ap.get_3d_distance_to(self.wifi_sta)
            self.ap_to_sta_phi, self.ap_to_sta_theta = self.wifi_ap.get_pointing_vector_to(
                self.wifi_sta,
            )

        # Get all currently active Access Points
        ap_active = np.where(self.wifi_ap.active)[0]
        
        # Iterate over each active Access Point
        for ap in ap_active:
            # Select K STA's among the ones that are connected to this AP
            random_number_gen.shuffle(self.link[ap])
            K = self.parameters.wifi.sta.k
            del self.wifi_link[ap][K:]
            
            # Activate the selected STA's and create beams if the AP is active
            if self.wifi_ap.active[ap]:
                self.wifi_sta.active[self.wifi_link[ap]] = np.ones(K, dtype=bool)
                
                for sta in self.wifi_link[ap]:
                    # Add a beam from the AP's antenna to the STA
                    self.wifi_ap.antenna[ap].add_beam(
                        self.ap_to_sta_phi[ap, sta],
                        self.ap_to_sta_theta[ap, sta],
                    )
                    
                    # Add a corresponding beam from the STA's antenna back to the AP
                    self.wifi_sta.antenna[sta].add_beam(
                        self.ap_to_sta_phi[ap, sta] - 180,
                        180 - self.ap_to_sta_theta[ap, sta],
                    )
                    
                    # Set beam resource block group for the STA
                    self.ap_to_sta_beam_rbs[sta] = len(
                        self.wifi_ap.antenna[ap].beams_list,
                    ) - 1
    
    def calculate_coupling_loss(self):
        # calculate the coupling loss between stations on active links
        pass


if __name__ == "__main__":
    wifi = SystemWifi(
        ParametersWifiSystem(),
        ParametersAntennaWifi(),
    )

    aps = wifi.generate_aps(np.random.RandomState(1))
    stas = wifi.generate_stas(np.random.RandomState(1))