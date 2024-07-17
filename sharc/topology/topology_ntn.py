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

    ALLOWED_NUM_SECTORS = [7, 19]

    def __init__(self, intersite_distance: float, cell_radius: int, bs_radius: float, bs_azimuth: float, bs_elevation: float, num_sectors=7):
        """
        Initializes the NTN topology with specific network settings.

        Parameters:
        intersite_distance: Distance between adjacent sites in meters.
        cell_radius: Radius of the coverage area for each site in meters.
        bs_radius: Distance of the base station (satellite) from the origin.
        bs_azimuth: Azimuth angle of the base station in degrees.
        bs_elevation: Elevation angle of the base station in degrees.
        num_sectors: Number of sectors for the topology (default is 7).
        """

        if num_sectors not in self.ALLOWED_NUM_SECTORS:
            raise ValueError(f"Invalid number of sectors: {num_sectors}. Allowed values are {self.ALLOWED_NUM_SECTORS}.")


        # Call to the superclass constructor to set common properties
        super().__init__(intersite_distance, cell_radius)
        self.bs_radius = bs_radius
        self.bs_azimuth = np.radians(bs_azimuth)
        self.bs_elevation = np.radians(bs_elevation)
        self.num_sectors = num_sectors

        # Calculate the base station coordinates
        self.bs_x = bs_radius * np.cos(self.bs_elevation) * np.cos(self.bs_azimuth)
        self.bs_y = bs_radius * np.cos(self.bs_elevation) * np.sin(self.bs_azimuth)
        self.bs_z = bs_radius * np.sin(self.bs_elevation)

        self.calculate_coordinates()

    def calculate_coordinates(self):
        """
        Computes the coordinates of each site. This is where the actual layout calculation would be implemented.
        """

        d = self.intersite_distance
        h = self.cell_radius

        self.x = [0]
        self.y = [0]

        # First ring (6 points)
        for k in range(6):
            angle = k * 60
            self.x.append(d * np.cos(np.radians(angle)))
            self.y.append(d * np.sin(np.radians(angle)))
        
        if self.num_sectors == 19:
            # Coordinates with 19 sectors
            # Second ring (12 points)
            for k in range(6):
                angle = k * 60
                self.x.append(2 * d * np.cos(np.radians(angle)))
                self.y.append(2 * d * np.sin(np.radians(angle)))
                self.x.append(d * np.cos(np.radians(angle)) + d * np.cos(np.radians(angle + 60)))
                self.y.append(d * np.sin(np.radians(angle)) + d * np.sin(np.radians(angle + 60)))


        self.x = np.array(self.x)
        self.y = np.array(self.y)

        self.z = np.zeros(len(self.x))  # Assuming all points are at ground level

        # Rotate the anchor points by 30 degrees
        theta = np.radians(30)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        self.x_rotated = self.x * cos_theta - self.y * sin_theta
        self.y_rotated = self.x * sin_theta + self.y * cos_theta

        # Calculate azimuth and elevation for each point
        self.azimuth = np.arctan2(self.y_rotated - self.bs_y, self.x_rotated - self.bs_x) * 180 / np.pi
        distance_xy = np.sqrt((self.x_rotated - self.bs_x)**2 + (self.y_rotated - self.bs_y)**2)
        self.elevation = np.arctan2(self.z - self.bs_z, distance_xy) * 180 / np.pi

        # Update the number of base stations after setup
        self.num_base_stations = len(self.x)
        self.indoor = np.zeros(self.num_base_stations, dtype=bool)

        # Update the number of base stations after setup
        self.num_base_stations = len(self.x)
        self.indoor = np.zeros(self.num_base_stations, dtype=bool)

        self.x = self.x_rotated
        self.y = self.y_rotated
        
    def plot(self, axis: matplotlib.axes.Axes):
        r = self.cell_radius

        # Plot each sector
        for x, y in zip(self.x, self.y):
            hexagon = []
            for a in range(6):
                angle_rad = math.radians(a * 60)
                hexagon.append([x + r * math.cos(angle_rad), y + r * math.sin(angle_rad)])
            hexagon.append(hexagon[0])  # Close the hexagon
            hexagon = np.array(hexagon)

            sector = plt.Polygon(hexagon, fill=None, edgecolor='k')
            axis.add_patch(sector)

        # Plot base stations
        axis.scatter(self.x, self.y, s=200, marker='v', c='k', edgecolor='k', linewidth=1, alpha=1,
                     label="Anchor Points")



# Example usage
if __name__ == '__main__':
    cell_radius = 100000  # meters
    intersite_distance = cell_radius * np.sqrt(3)  # meters
    bs_radius = 20000  # meters
    bs_azimuth = 45  # degrees
    bs_elevation = 90  # degrees

    # For 7 sectors
    ntn_topology_7 = TopologyNTN(
        intersite_distance, cell_radius, bs_radius, bs_azimuth, bs_elevation, num_sectors=7)
    ntn_topology_7.calculate_coordinates()  # Calculate the site coordinates

    print(ntn_topology_7.azimuth)
    print(ntn_topology_7.elevation)

    fig, ax = plt.subplots()
    ntn_topology_7.plot(ax)  # Plot the topology

    plt.axis('image')
    name = "NTN System - 1 Cluster with 7 sectors"
    plt.title(name)

    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # For 19 sectors
    ntn_topology_19 = TopologyNTN(
        intersite_distance, cell_radius, bs_radius, bs_azimuth, bs_elevation, num_sectors=19)
    ntn_topology_19.calculate_coordinates()  # Calculate the site coordinates

    print(ntn_topology_19.azimuth)
    print(ntn_topology_19.elevation)

    fig, ax = plt.subplots()
    ntn_topology_19.plot(ax)  # Plot the topology

    plt.axis('image')
    name = "NTN System - 1 Cluster with 19 sectors"
    plt.title(name)

    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    