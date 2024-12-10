import os
import sys

from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.simulation_downlink import SimulationDownlink
from sharc.station_manager import StationManager
from sharc.topology.topology_hotspot import TopologyHotspot


class SystemWifi():
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem) -> None:
        self.parameters = param
        self.topology = self.generate_topology(self.parameters)
        
    '''def initialize_simulations(self):
        param_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input', 'parameters.yaml')
        simulation_dl = SimulationDownlink(self.parameters, param_file)
        simulation_ul = SimulationUplink(self.parameters, param_file)
        return (simulation_dl, simulation_ul)'''

    def generate_topology(self, parameters):
        if self.parameters.topology.type == "HOTSPOT":
            return TopologyHotspot(
                parameters.topology.hotspot,
                parameters.topology.hotspot.intersite_distance,
                parameters.topology.hotspot.num_clusters
            )
        sys.stderr.write(
            "ERROR\nInvalid topology: " +
            parameters.topology.type,
        )
        sys.exit(1)

    def generate_stations(self):
        # Generate the stations similar to the StationFactory functions. 
        # The wifi stations MUST be Station Manager objects
        pass

    def connect_stations(self):
        # Create a boolean list that list which links between tx and rx are active
        pass

    def calculate_coupling_loss(self):
        # calculate the coupling loss between stations on active links
        pass

    def calculate_gains(self):
        # calculate the gains between stations on active links
        pass
