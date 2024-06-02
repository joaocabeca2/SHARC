from sharc.topology.topology import Topology
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes

class TopologyNTN(Topology):
    """
    Class to generate and manage the topology of Non-Terrestrial Network (NTN) sites
    based on a specified macrocell network topology.
    """

    # Define allowed configurations for validation purposes
    
    ALLOWED_NUM_SECTORS = [1, 3, 7]  # Supported sector configurations

    def __init__(self, intersite_distance: float, cell_radius: int, bs_height: float,  azimuth_ntn: np.array, elevation_ntn: np.array):
        """
        Initializes the NTN topology with specific network settings.

        Parameters:
        intersite_distance: Distance between adjacent sites in meters.
        cell_radius: Radius of the coverage area for each site in meters.
        bs_height: Altitude of the base station in meters.
      
        azimuth_ntn: Array of azimuth angles for each sector, defining the horizontal alignment.
        elevation_ntn: Array of elevation angles for each sector, defining the vertical alignment.
        """


        if len(azimuth_ntn) not in self.ALLOWED_NUM_SECTORS:
            raise ValueError(f"Number of sectors ({len(azimuth_ntn)}) not allowed. Allowed values are {self.ALLOWED_NUM_SECTORS}.")

        if len(azimuth_ntn) != len(elevation_ntn):
            raise ValueError("Azimuth and elevation arrays must have the same length.")

        # Call to the superclass constructor to set common properties
        super().__init__(intersite_distance, cell_radius)
        self.is_space_station = True
        self.bs_height = bs_height
        self.azimuth = azimuth_ntn
        self.elevation = elevation_ntn
        self.space_station_x = None
        self.space_station_y = None
        self.num_sectors = len(azimuth_ntn)  # Derive the number of sectors from the length of the azimuth array

    
    def calculate_coordinates(self,random_number_gen=np.random.RandomState()):
        """
        Computes the coordinates of each site. This is where the actual layout calculation would be implemented.
        """

        d = self.intersite_distance
        h = self.cell_radius

        # Set coordinates for a single cluster, potentially extending this for multiple clusters (Only for 7 beams)
        self.x = np.array([0, d, -d, d/2, -d/2, d/2, -d/2])
        self.y = np.array([0, 0, 0, ((d/np.sqrt(3)) + h/2), -((d/np.sqrt(3)) + h/2), -((d/np.sqrt(3)) + h/2), ((d/np.sqrt(3)) + h/2)])

        self.space_station_x = np.zeros(len(self.x))
        self.space_station_y = np.zeros(len(self.y))

        # Update the number of base stations after setup
        self.num_base_stations = len(self.x)
        self.indoor = np.zeros(self.num_base_stations, dtype=bool)

    def plot(self, axis: matplotlib.axes.Axes):
        # create the hexagons for 1 Sectors
        global azimuth, r, angle
        if self.num_sectors == 1:
            r = self.cell_radius
            azimuth = np.zeros(self.x.size)

        # create the hexagons for 3 Sectors
        elif self.num_sectors == 3:
            r = self.intersite_distance / np.sqrt(3)
            azimuth = self.azimuth

        # create the hexagon for 7 Sectors
        elif self.num_sectors == 7:
            r = self.cell_radius
            azimuth = np.zeros(self.x.size)

        # create the dodecagon for 19 Sectors
        elif self.num_sectors == 19:
            r = self.intersite_distance / np.sqrt(3)
            azimuth = np.zeros(self.x.size)

        # create the hexagons for 3 Sectors
        for x, y, az in zip(self.x, self.y, azimuth):

            if self.num_sectors == 1:
                x = x - self.intersite_distance / 2
                y = y - r / 2
                angle = int(az - 30)

            elif self.num_sectors == 3:
                angle = int(az - 30)

        # create the hexagon for 7 Sectors
            
            if self.num_sectors == 7:
                x = x - self.intersite_distance / 2
                y = y - r / 2
                # y = y - self.intersite_distance
                angle = int(az - 30)

            

            se = list([[x, y]])

            # plot polygon - 1 Sectors
            if self.num_sectors == 1:
                for a in range(7):
                    se.extend(
                        [[se[-1][0] + r * math.cos(math.radians(angle)), se[-1][1] +
                          r * math.sin(math.radians(angle))]])
                    angle += 60
                sector = plt.Polygon(se, fill=None, edgecolor='k')
                axis.add_patch(sector)

                # plot polygon - 3 Sectors
            elif self.num_sectors == 3:
                for a in range(6):
                    se.extend(
                        [[se[-1][0] + r / 2 * math.cos(math.radians(angle)), se[-1][1] +
                          r / 2 * math.sin(math.radians(angle))]])
                    angle += 60
                sector = plt.Polygon(
                    se, fill=None, edgecolor='blue', linewidth=1, alpha=1)
                axis.add_patch(sector)

            # plot polygon - 7 Sectors
            elif self.num_sectors == 7:
                for a in range(7):
                    se.extend(
                        [[se[-1][0] + r * math.cos(math.radians(angle)), se[-1][1] +
                          r * math.sin(math.radians(angle))]])
                    angle += 60
                sector = plt.Polygon(se, fill=None, edgecolor='k')
                axis.add_patch(sector)

                # plot polygon - 19 Sectors
            elif self.num_sectors == 19:
                for a in range(25):
                    se.extend(
                        [[se[0][0] + r * math.cos(math.radians(angle)),
                          se[0][0] + r * math.sin(math.radians(angle))]])
                    angle += 15
                sector = plt.Polygon(se[::-2], fill=None, edgecolor='k')
                axis.add_patch(sector)

        # macro cell base stations
        axis.scatter(self.x, self.y, s=200, marker='v', c='k', edgecolor='k', linewidth=1, alpha=1,
                     label="Anchor Points")
      

# Example usage
if __name__ == '__main__':
    cell_radius = 100000  # meters
    intersite_distance = cell_radius * np.sqrt(3)  # meters
    bs_height = 20000  # meters
    num_clusters = 1  # number of clusters

    # Three Sectors

    #azimuth_ntn = np.array([60,180,300])
    #elevation_ntn = np.array([-90,-90,-90])

    # Seven Sectors
    azimuth_ntn = np.array([0,0,60,120,180,240,300])
    elevation_ntn = np.array([-90,-23,-23,-23,-23,-23,-23])


    ntn_topology = TopologyNTN(intersite_distance, cell_radius, bs_height, azimuth_ntn, elevation_ntn)
    ntn_topology.calculate_coordinates()  # Calculate the site coordinates

    fig, ax = plt.subplots()
    ntn_topology.plot(ax)  # Plot the topology
    
    plt.axis('image')
    name = "NTN System - {} Cluster{} with {} sectors".format(num_clusters,"s" if num_clusters != 1 else "",  # handles pluralization
    len(azimuth_ntn)
    )
    plt.title(name)

    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")
    plt.legend()
    plt.tight_layout()
    plt.show()
