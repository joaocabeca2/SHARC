
"""
This module implements an IMT Mobile Satellite Service (MSS) for Direct Connectivity (D2D) topology.

It consists of a group of NGSO satellites that provide direct connectivity to user equipment (UE) on the ground.
The Space Stations positions are generated from the Keplerian elements of the orbits in the OrbitModel class.
Only a subset of Space Stations are used, which are the ones that are visible to the UE.
After satellite visibility is calculated, the ECEF coordinates are transformed to a new cartesian coordinate system
centered at the reference point defined in the GeometryConverter object.
The azimuth and elevation angles are also rotated to the new coordinate system.
The visible Space Stations are then used to generate the IMT Base Stations.
"""

import numpy as np

from sharc.topology.topology import Topology
from sharc.parameters.imt.parameters_imt_mss_dc import ParametersImtMssDc
from sharc.parameters.parameters_orbit import ParametersOrbit
from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.support.sharc_geom import GeometryConverter
from sharc.satellite.utils.sat_utils import calc_elevation
from sharc.support.sharc_geom import lla2ecef, cartesian_to_polar, polar_to_cartesian


class TopologyImtMssDc(Topology):
    def __init__(self, params: ParametersImtMssDc, geometry_converter: GeometryConverter):
        """Implements a IMT Mobile Satellite Service (MSS) for Direct Connectivity (D2D) topology.

        Parameters
        ----------
        params : ParametersImtMssDc
            Input parameters for the IMT MSS-DC topology
        geometry_converter : GeometryConverter
            GeometryConverter object that converts the ECEF coordintate system to one
            centered at GeometryConverter.reference.
        """
        self.is_space_station = True
        self.num_sectors = params.num_beams

        # Specific attributes
        self.geometry_converter = geometry_converter
        self.orbit_params = params
        self.lat = None
        self.lon = None
        self.min_elev_angle = params.min_elev_angle  # Minimum elevation angle for satellite visibility
        self.orbits = []

        # Iterate through each orbit defined in the parameters
        for param in self.orbit_params.orbits:
            # Instantiate an OrbitModel for the current orbit
            orbit = OrbitModel(
                Nsp=param.sats_per_plane,  # Satellites per plane
                Np=param.n_planes,  # Number of orbital planes
                phasing=param.phasing_deg,  # Phasing angle in degrees
                long_asc=param.long_asc_deg,  # Longitude of ascending node in degrees
                omega=param.omega_deg,  # Argument of perigee in degrees
                delta=param.inclination_deg,  # Orbital inclination in degrees
                hp=param.perigee_alt_km,  # Perigee altitude in kilometers
                ha=param.apogee_alt_km,  # Apogee altitude in kilometers
                Mo=param.initial_mean_anomaly  # Initial mean anomaly in degrees
            )
            self.orbits.append(orbit)

    def calculate_coordinates(self, random_number_gen=np.random.RandomState()):
        """
        Computes the coordintates of the visible space stations
        """
        self.geometry_converter.validate()

        # Calculate the total number of satellites across all orbits
        total_satellites = sum(orbit.n_planes * orbit.sats_per_plane for orbit in self.orbit_params.orbits)

        idx_orbit = np.zeros(total_satellites, dtype=int)  # Add orbit index array

        # Initialize arrays to store satellite positions, angles and distance from center of earth
        all_positions = {"R": [], "lat": [], "lon": [], "sx": [], "sy": [], "sz": []}
        all_elevations = []  # Store satellite elevations
        all_azimuths = []  # Store satellite azimuths

        # List to store indices of active satellites
        active_satellite_idxs = []
        current_sat_idx = 0  # Index tracker for satellites across all orbits

        MAX_ITER = 100  # Maximum iterations to find at least one visible satellite
        i = 0  # Iteration counter for ensuring satellite visibility
        while len(active_satellite_idxs) == 0:
            # Iterate through each orbit defined in the parameters
            for orbit_idx, orbit in enumerate(self.orbits):
                # Generate random positions for satellites in this orbit
                pos_vec = orbit.get_orbit_positions_random_time(rng=random_number_gen)

                # Determine the number of satellites in this orbit
                num_satellites = len(pos_vec["sx"])

                # Assign orbit index to satellites
                idx_orbit[current_sat_idx:current_sat_idx + num_satellites] = orbit_idx

                # Extract satellite positions and calculate distances
                sx, sy, sz = pos_vec['sx'], pos_vec['sy'], pos_vec['sz']
                r = np.sqrt(sx**2 + sy**2 + sz**2)  # Distance from Earth's center

                # When getting azimuth and elevation, we need to consider sx, sy and sz points
                # from the center of earth to the satellite, and we need to point the satellite
                # towards the center of earth
                elevations = np.degrees(np.arcsin(-sz / r))  # Calculate elevation angles
                azimuths = np.degrees(np.arctan2(-sy, -sx))  # Calculate azimuth angles

                # Append satellite positions and angles to global lists
                all_positions['lat'].extend(pos_vec['lat'])  # Latitudes
                all_positions['lon'].extend(pos_vec['lon'])  # Longitudes
                all_positions['sx'].extend(sx)  # X-coordinates
                all_positions['sy'].extend(sy)  # Y-coordinates
                all_positions['sz'].extend(sz)  # Z-coordinates
                all_positions["R"].extend(r)
                all_elevations.extend(elevations)  # Elevation angles
                all_azimuths.extend(azimuths)  # Azimuth angles

                # Calculate satellite visibility from base stations
                elev_from_bs = calc_elevation(
                    self.geometry_converter.ref_lat,  # Latitude of base station
                    pos_vec['lat'],  # Latitude of satellites
                    self.geometry_converter.ref_long,  # Longitude of base station
                    pos_vec['lon'],  # Longitude of satellites
                    orbit.perigee_alt_km  # Perigee altitude in kilometers
                )

                # Determine visible satellites based on minimum elevation angle
                visible_sat_idxs = [
                    current_sat_idx + idx for idx, elevation in enumerate(elev_from_bs)
                    if elevation >= self.min_elev_angle
                ]
                active_satellite_idxs.extend(visible_sat_idxs)

                # Update the index tracker for the next orbit
                current_sat_idx += len(sx)

            i += 1  # Increment iteration counter
            if i >= MAX_ITER:  # Check if maximum iterations reached
                raise RuntimeError(
                    "Maximum iterations reached, and no satellite was selected within the minimum elevation criteria."
                )
        # We have the list of visible satellites, now create a Topolgy of this subset and move the coordinate system
        # reference.
        self.x = np.squeeze(np.array(all_positions['sx']))[active_satellite_idxs] * 1e3  # Convert X-coordinates to meters
        self.y = np.squeeze(np.array(all_positions['sy']))[active_satellite_idxs] * 1e3  # Convert Y-coordinates to meters
        self.z = np.squeeze(np.array(all_positions['sz']))[active_satellite_idxs] * 1e3  # Convert Z-coordinates to meters
        self.elevation = np.squeeze(np.array(all_elevations))[active_satellite_idxs]  # Elevation angles
        self.azimuth = np.squeeze(np.array(all_azimuths))[active_satellite_idxs]  # Azimuth angles
        self.num_base_stations = len(self.x)
        self.indoor = np.zeros(self.num_base_stations, dtype=bool)

        # Store the latitude and longitude of the visible satellites for later use
        self.lat = np.squeeze(np.array(all_positions['lat']))[active_satellite_idxs]
        self.lon = np.squeeze(np.array(all_positions['lon']))[active_satellite_idxs]

        # Convert the ECEF coordinates to the transformed cartesian coordinates and set the Space Station positions
        # used to generetate the IMT Base Stations
        self.x, self.y, self.z = \
            self.geometry_converter.convert_cartesian_to_transformed_cartesian(self.x, self.y, self.z)

        # Rotate the azimuth and elevation angles off the center beam the new transformed cartesian coordinates
        r = 1
        # transform pointing vectors, without considering geodesical earth coord system
        pointing_vec_x, pointing_vec_y, pointing_vec_z = polar_to_cartesian(r, self.azimuth, self.elevation)
        pointing_vec_x, pointing_vec_y, pointing_vec_z = \
            self.geometry_converter.convert_cartesian_to_transformed_cartesian(
                pointing_vec_x, pointing_vec_y, pointing_vec_z, translate=0)
        _, self.azimuth, self.elevation = cartesian_to_polar(pointing_vec_x, pointing_vec_y, pointing_vec_z)


# Example usage
if __name__ == '__main__':
    from sharc.parameters.imt.parameters_imt_mss_dc import ParametersImtMssDc
    from sharc.support.sharc_geom import GeometryConverter

    # Define the parameters for the IMT MSS-DC topology
    # SystemA Orbit parameters
    orbit = ParametersOrbit(
        n_planes=20,
        sats_per_plane=32,
        phasing_deg=3.9,
        long_asc_deg=18.0,
        inclination_deg=54.5,
        perigee_alt_km=525,
        apogee_alt_km=525
    )
    params = ParametersImtMssDc(
        beam_radius=36516.0,
        num_beams=19,
        orbits=[orbit]
    )

    # Define the geometry converter
    geometry_converter = GeometryConverter()
    geometry_converter.set_reference(-15.0, -42.0, 1200)

    # Instantiate the IMT MSS-DC topology
    imt_mss_dc_topology = TopologyImtMssDc(params, geometry_converter)

    # Calculate the coordinates of the space stations
    rng = np.random.RandomState(101)
    imt_mss_dc_topology.calculate_coordinates(random_number_gen=rng)

    # Plot the IMT MSS-DC space stations after selecting the visible ones and transforming the coordinate system
    import plotly.graph_objects as go

    # Create a 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=imt_mss_dc_topology.x / 1e3,
        y=imt_mss_dc_topology.y / 1e3,
        z=imt_mss_dc_topology.z / 1e3,
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.8
        )
    )])

    # Set plot title and axis labels
    fig.update_layout(
        title='IMT MSS-DC Topology',
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]'
        )
    )

    # Add a point at the origin
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=1.0
        ),
        name='Reference'
    ))
    # Calculate the elevation with respect to the x-y plane
    elevation_xy_plane = np.degrees(
        np.arctan2(
            imt_mss_dc_topology.z,
            np.sqrt(imt_mss_dc_topology.x**2 + imt_mss_dc_topology.y**2)
        )
    )

    # Add the elevation with respect to the x-y plane to the plot
    fig.add_trace(go.Scatter3d(
        x=imt_mss_dc_topology.x / 1e3,
        y=imt_mss_dc_topology.y / 1e3,
        z=imt_mss_dc_topology.z / 1e3,
        mode='markers',
        marker=dict(
            size=4,
            color=elevation_xy_plane,
            colorscale='Viridis',
            colorbar=dict(title='Elevation (degrees)', x=-0.1),
            opacity=0.8
        ),
        name='Space Stations Positions'
    ))

    # Add lines between the origin and the IMT space stations
    for x, y, z in zip(imt_mss_dc_topology.x / 1e3, imt_mss_dc_topology.y / 1e3, imt_mss_dc_topology.z / 1e3):
        fig.add_trace(go.Scatter3d(
            x=[0, x],
            y=[0, y],
            z=[0, z],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Elevation Line'
        ))

    # Plot boresights
    boresight_length = 100  # Length of the boresight vectors for visualization
    boresight_x, boresight_y, boresight_z = polar_to_cartesian(
        boresight_length,
        imt_mss_dc_topology.azimuth,
        imt_mss_dc_topology.elevation
    )
    # Add arrow heads to the end of the boresight vectors
    for x, y, z, bx, by, bz in zip(imt_mss_dc_topology.x / 1e3,
                                   imt_mss_dc_topology.y / 1e3,
                                   imt_mss_dc_topology.z / 1e3,
                                   boresight_x,
                                   boresight_y,
                                   boresight_z):
        fig.add_trace(go.Cone(
            x=[x + bx],
            y=[y + by],
            z=[z + bz],
            u=[bx],
            v=[by],
            w=[bz],
            colorscale=[[0, 'orange'], [1, 'orange']],
            sizemode='absolute',
            sizeref=40,
            showscale=False
        ))
    for x, y, z, bx, by, bz in zip(imt_mss_dc_topology.x / 1e3,
                                   imt_mss_dc_topology.y / 1e3,
                                   imt_mss_dc_topology.z / 1e3,
                                   boresight_x,
                                   boresight_y,
                                   boresight_z):
        fig.add_trace(go.Scatter3d(
            x=[x, x + bx],
            y=[y, y + by],
            z=[z, z + bz],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Boresight'
        ))

    # Maintain axis proportions
    fig.update_layout(scene_aspectmode='data')

    fig.show()

    # Print the elevation angles
    print('Elevation angles:', imt_mss_dc_topology.elevation)
    # Print the azimuth angles
    print('Azimuth angles:', imt_mss_dc_topology.azimuth)
    # Print the elevation w.r.t. the x-y plane
    print('Elevation w.r.t. XY plane:', elevation_xy_plane)
    # Print the slant range
    print('Slant range:', np.sqrt(imt_mss_dc_topology.x**2 + imt_mss_dc_topology.y**2 + imt_mss_dc_topology.z**2) / 1e3)
