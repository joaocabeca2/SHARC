from sharc.parameters.parameters_wifi_system import ParametersWifiSystem
from sharc.station_manager import StationManager

class SystemWifi():
    """Implements a Wifi Network compose of APs and STAs."""
    def __init__(self, param: ParametersWifiSystem) -> None:
        # initialize the parameters of the wifi system
        pass

    def generate_topology(self):
        # Create the positions of the Wifi stations
        pass

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
