import math
import sys

import numpy as np
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.parameters.wifi.parameters_indoor import ParametersIndoor
#om sharc.propagation.propagation_factory import PropagationFactory
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.topology.topology import Topology
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi

from itertools import product

class TopologyIndoor(Topology):
    """
    Generates the coordinates of the sites based on the indoor network
    topology.
    """
     
    def __init__(self, param: ParametersIndoor):
        """
        Constructor method that sets the parameters.

        Parameters
        ----------
            param : parameters of the indoor topology
        """

        # These are the building's width, deep and height
        # They do not change
        self.b_w = 120
        self.b_d = 50
        self.b_h = 3

        cell_radius = param.intersite_distance / 2
        super().__init__(param.intersite_distance, cell_radius)

        self.n_rows = param.n_rows
        self.n_colums = param.n_colums
        self.street_width = param.street_width
        self.sta_indoor_percent = param.sta_indoor_percent
        self.building_class = param.building_class
        self.num_cells = param.num_cells
        self.num_floors = param.num_floors
        if param.num_wifi_buildings == 'ALL':
            self.all_buildings = True
            self.num_wifi_buildings = self.n_rows * self.n_colums
        else:
            self.all_buildings = False
            self.num_wifi_buildings = int(param.num_wifi_buildings)
        self.wifi_buildings = list()
        self.total_ap_level = self.num_wifi_buildings * self.num_cells

        self.height = np.empty(0)

    def calculate_coordinates(self, random_number_gen=np.random.RandomState()):
        """
        Calculates the coordinates of the stations according to the inter-site
        distance parameter. This method is invoked in all snapshots but it can
        be called only once for the indoor topology. So we set
        static_base_stations to True to avoid unnecessary calculations.
        """
        if not self.static_base_stations:
            self.reset()
            self.static_base_stations = self.all_buildings

            x_base = np.array(
                [(2 * k + 1) * self.cell_radius for k in range(self.num_cells)],
            )
            y_base = self.b_d / 2 * np.ones(self.num_cells)

            # Choose random buildings
            all_buildings = list(
                product(range(self.n_rows), range(self.n_colums)),
            )
            random_number_gen.shuffle(all_buildings)
            self.wifi_buildings = all_buildings[:self.num_wifi_buildings]

            floor_x = np.empty(0)
            floor_y = np.empty(0)
            for build in self.wifi_buildings:
                r = build[0]
                c = build[1]
                floor_x = np.concatenate(
                    (floor_x, x_base + c * (self.b_w + self.street_width)),
                )
                floor_y = np.concatenate(
                    (floor_y, y_base + r * (self.b_d + self.street_width)),
                )

            for f in range(self.num_floors):
                self.x = np.concatenate((self.x, floor_x))
                self.y = np.concatenate((self.y, floor_y))
                self.height = np.concatenate((
                    self.height,
                    (f + 1) * self.b_h * np.ones_like(floor_x),
                ))

            # In the end, we have to update the number of base stations
            self.num_base_stations = len(self.x)

            self.azimuth = np.zeros(self.num_base_stations)
            self.indoor = np.ones(self.num_base_stations, dtype=bool)
        
    def reset(self):
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.height = np.empty(0)
        self.azimuth = np.empty(0)
        self.indoor = np.empty(0)
        self.num_base_stations = -1
        self.static_base_stations = False


class SystemWifi:
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem, param_ant_ap: ParametersAntennaWifi, param_ant_sta: ParametersAntennaWifi, random_number_gen: np.random.RandomState) -> None:
        self.parameters = param
        self.parameters_antenna = param_ant_ap
        self.topology = TopologyIndoor(self.parameters.topology.indoor) 
        self.topology.calculate_coordinates()
        self.num_aps = self.topology.num_base_stations
        self.num_sta = self.num_aps * self.parameters.sta.k * self.parameters.sta.k_m


        self.wrap_around_enabled = False

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
        self.path_loss = np.empty([self.num_aps, self.num_sta])
        self.coupling_loss = np.empty([self.num_aps, self.num_sta])

        self.ap_to_sta_phi = np.empty([self.num_aps, self.num_sta])
        self.ap_to_sta_theta = np.empty([self.num_aps, self.num_sta])
        self.ap_to_sta_beam_rbs = -1.0 * np.ones(self.num_sta, dtype=int)

        self.sta = np.empty(self.num_sta)
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

        self.inr = np.empty([self.num_aps, self.num_sta])
        self.rx_interference = np.empty(0)

        self.ap = self.generate_aps(random_number_gen)
        self.sta = self.generate_stas(random_number_gen)
    
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
            self.topology.b_w / math.sqrt(self.topology.sta_indoor_percent) - self.topology.b_w
        ) / 2
        delta_y = (
            self.topology.b_d / math.sqrt(self.topology.sta_indoor_percent) - self.topology.b_d
        ) / 2

        for ap in range(self.num_aps):
            idx = [
                i for i in range(
                    ap * num_sta_per_ap, ap * num_sta_per_ap + num_sta_per_ap,
                )
            ]
            if ap % self.topology.num_cells == 0 and ap < self.topology.total_ap_level:
                x_min = self.topology.x[ap] - self.topology.cell_radius - delta_x
                x_max = self.topology.x[ap] + self.topology.cell_radius
            elif ap % self.topology.num_cells == self.topology.num_cells - 1 and ap < self.topology.total_ap_level:
                x_min = self.topology.x[ap] - self.topology.cell_radius
                x_max = self.topology.x[ap] + self.topology.cell_radius + delta_x
            else:
                x_min = self.topology.x[ap] - self.topology.cell_radius
                x_max = self.topology.x[ap] + self.topology.cell_radius

            if ap < self.topology.total_ap_level:
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
                wifi_sta.indoor[idx] = ~out

        wifi_sta.x = np.array(sta_x)
        wifi_sta.y = np.array(sta_y)
        wifi_sta.height = np.array(sta_z)

        wifi_sta.active = np.zeros(self.num_sta, dtype=bool)
        wifi_sta.rx_interference = -500 * np.ones(self.num_sta)
        wifi_sta.ext_interference = -500 * np.ones(self.num_sta)
        wifi_sta.bandwidth = self.parameters.bandwidth * np.ones(self.num_sta)
        wifi_sta.center_freq = self.parameters.frequency * np.ones(self.num_sta)
        wifi_sta.noise_figure = self.parameters.ap.noise_figure * np.ones(self.num_sta)
        wifi_sta.thermal_noise = -500 * np.ones(self.num_sta)

        for i in range(self.num_sta):
            wifi_sta.antenna[i] = AntennaOmni()

        return wifi_sta
    
    def connect_wifi_sta_to_ap(self, parameters: ParametersWifiSystem):
        """
        Link the Wi-Fi STA's to the serving AP. It is assumed that each group of K
        user equipments are distributed and pointed to a certain access point
        """
        num_sta_per_ap = parameters.sta.k * parameters.sta.k_m
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            sta_list = [
                i for i in range(
                    ap * num_sta_per_ap, ap * num_sta_per_ap + num_sta_per_ap,
                )
            ]
            self.link[ap] = sta_list

    def select_sta(self, random_number_gen: np.random.RandomState, parameters: ParametersWifiSystem):
        """
        Select K STAs randomly from all the STAs linked to one AP as “chosen”
        STAs. These K “chosen” STAs will be scheduled during this snapshot.
        """
        # Calculate distances and angles between Access Points (APs) and Stations (STAs)
        if self.wrap_around_enabled:
            self.ap_to_sta_d_2D, self.ap_to_sta_d_3D, self.ap_to_sta_phi, self.ap_to_sta_theta = \
                self.ap.get_dist_angles_wrap_around(self.sta)
        else:
            self.ap_to_sta_d_2D = self.ap.get_distance_to(self.sta)
            self.ap_to_sta_d_3D = self.ap.get_3d_distance_to(self.sta)
            self.ap_to_sta_phi, self.ap_to_sta_theta = self.ap.get_pointing_vector_to(
                self.sta,
            )

        # Get all currently active Access Points
        ap_active = np.where(self.ap.active)[0]
        
        # Iterate over each active Access Point
        for ap in ap_active:
            # Select K STA's among the ones that are connected to this AP
            random_number_gen.shuffle(self.link[ap])
            K = parameters.sta.k
            del self.link[ap][K:]
            
            # Activate the selected STA's and create beams if the AP is active
            if self.ap.active[ap]:
                self.sta.active[self.link[ap]] = np.ones(K, dtype=bool)
                
                for sta in self.link[ap]:
                    # Add a beam from the AP's antenna to the STA
                    self.ap.antenna[ap].add_beam(
                        self.ap_to_sta_phi[ap, sta],
                        self.ap_to_sta_theta[ap, sta],
                    )
                    
                    # Add a corresponding beam from the STA's antenna back to the AP
                    self.sta.antenna[sta].add_beam(
                        self.ap_to_sta_phi[ap, sta] - 180,
                        180 - self.ap_to_sta_theta[ap, sta],
                    )
                    
                    # Set beam resource block group for the STA
                    self.ap_to_sta_beam_rbs[sta] = len(
                        self.ap.antenna[ap].beams_list,
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