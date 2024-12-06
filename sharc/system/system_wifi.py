from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.topology.topology_hotspot import TopologyHotspot
from sharc.mask.spectral_mask_3gpp import SpectralMask3Gpp
import sys
import numpy as np

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

    def generate_stations(self, random_number_gen: np.random.RandomState):
        param_ant = self.parameters.ap.antenna
        num_aps = self.topology.num_base_stations
        #num_stas = num_aps * self.parameters
        access_points = StationManager(num_aps)
        access_points.station_type = StationType.WIFI_SYSTEM

        access_points.x = self.topology.x
        access_points.y = self.topology.y
        access_points.elevation = -param_ant.downtilt * np.ones(num_aps)
        access_points.height = self.parameters.ap.height
        
        access_points.azimuth = self.topology.azimuth
        access_points.active = random_number_gen.rand(
            num_aps,
        ) < self.parameters.ap.load_probability
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

    def connect_stations(self):
        # Create a boolean list that list which links between tx and rx are active
        pass

    def calculate_coupling_loss(self):
        # calculate the coupling loss between stations on active links
        pass

    def calculate_gains(self):
        # calculate the gains between stations on active links
        pass
