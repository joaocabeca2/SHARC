# Generates a 3D plot of the Earth with the satellites positions
# https://geopandas.org/en/stable/docs/user_guide/io.html
import os
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.satellite.utils.sat_utils import calc_elevation, lla2ecef
from sharc.satellite.ngso.constants import EARTH_RADIUS_KM
from sharc.parameters.parameters_mss_d2d import ParametersOrbit, ParametersMssD2d
from sharc.station_factory import StationFactory


def plot_back(fig):
    """back half of sphere"""
    clor = 'rgb(220, 220, 220)'
    R = np.sqrt(EARTH_RADIUS_KM)
    u_angle = np.linspace(0, np.pi, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(R * np.cos(u_angle), R * np.sin(v_angle))
    y_dir = np.outer(R * np.sin(u_angle), R * np.sin(v_angle))
    z_dir = np.outer(R * np.ones(u_angle.shape[0]), R * np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]],
                    opacity=1.0, showlegend=False, lighting=dict(
                    # opacity=fig.sphere_alpha, colorscale=[[0, fig.sphere_color], [1, fig.sphere_color]])
                    diffuse=0.1))


def plot_front(fig):
    """front half of sphere"""
    clor = 'rgb(220, 220, 220)'
    R = np.sqrt(EARTH_RADIUS_KM)
    u_angle = np.linspace(-np.pi, 0, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(R * np.cos(u_angle), R * np.sin(v_angle))
    y_dir = np.outer(R * np.sin(u_angle), R * np.sin(v_angle))
    z_dir = np.outer(R * np.ones(u_angle.shape[0]), R * np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False,
                    lighting=dict(
                        # opacity=fig.sphere_alpha, colorscale=[[0, fig.sphere_color], [1, fig.sphere_color]])
                        diffuse=0.1))


def plot_polygon(poly):

    xy_coords = poly.exterior.coords.xy
    lon = np.array(xy_coords[0])
    lat = np.array(xy_coords[1])

    lon = lon * np.pi / 180
    lat = lat * np.pi / 180

    R = EARTH_RADIUS_KM
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return x, y, z


def plot_globe_with_borders():
    # Read the shapefile.  Creates a DataFrame object
    countries_borders_shp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "../data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    gdf = gpd.read_file(countries_borders_shp_file)
    fig = go.Figure()
    plot_front(fig)
    plot_back(fig)

    for i in gdf.index:
        # print(gdf.loc[i].NAME)            # Call a specific attribute

        polys = gdf.loc[i].geometry         # Polygons or MultiPolygons

        if polys.geom_type == 'Polygon':
            x, y, z = plot_polygon(polys)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                       line=dict(color='rgb(0, 0,0)'), showlegend=False))

        elif polys.geom_type == 'MultiPolygon':

            for poly in polys.geoms:
                x, y, z = plot_polygon(poly)
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                           line=dict(color='rgb(0, 0,0)'), showlegend=False))
    return fig


if __name__ == "__main__":
    # Define the orbit parameters for two satellite constellations
    orbit_1 = ParametersOrbit(
        n_planes=6, sats_per_plane=8, phasing_deg=7.5, long_asc_deg=0,
        inclination_deg=52, perigee_alt_km=1414, apogee_alt_km=1414
    )

    orbit_2 = ParametersOrbit(
        n_planes=4, sats_per_plane=10, phasing_deg=5.0, long_asc_deg=90,
        inclination_deg=60, perigee_alt_km=1200, apogee_alt_km=1200
    )

    # Configure the MSS D2D system parameters
    params = ParametersMssD2d(
        name="Example-MSS-D2D",
        antenna_pattern="ITU-R-S.1528-Taylor",
        antenna_gain=30.0,
        orbits=[orbit_1, orbit_2]
    )

    # Create a topology with a single base station
    from sharc.topology.topology_single_base_station_spherical import TopologySingleBaseStationSpherical
    imt_topology = TopologySingleBaseStationSpherical(
        cell_radius=500, num_clusters=1, central_latitude=-15.7801, central_longitude=-47.9292
    )

    # Create a random number generator
    rng = np.random.RandomState(seed=42)

    # Number of iterations (drops)
    NUM_DROPS = 100

    # Lists to store satellite positions (all and visible)
    all_positions = {'x': [], 'y': [], 'z': []}
    visible_positions = {'x': [], 'y': [], 'z': []}

    # Plot the ground station (blue marker)
    ground_sta_pos = lla2ecef(-15.7801, -42.9292, 0.0)

    for _ in range(NUM_DROPS):
        # Generate satellite positions using the StationFactory
        mss_d2d_manager = StationFactory.generate_mss_d2d_multiple_orbits(params, rng, imt_topology)

        # Extract satellite positions
        x_vec = mss_d2d_manager.x /1e3 #(Km)
        y_vec = mss_d2d_manager.y /1e3 #(Km)
        z_vec = mss_d2d_manager.height /1e3 #(Km) 
        # Store all positions
        all_positions['x'].extend(x_vec)
        all_positions['y'].extend(y_vec)
        all_positions['z'].extend(z_vec)

        # Identify visible satellites
        vis_sat_idxs = np.where(mss_d2d_manager.active)[0]
        visible_positions['x'].extend(x_vec[vis_sat_idxs])
        visible_positions['y'].extend(y_vec[vis_sat_idxs])
        visible_positions['z'].extend(z_vec[vis_sat_idxs])

    # Flatten arrays
    all_positions['x'] = np.concatenate(all_positions['x'])
    all_positions['y'] = np.concatenate(all_positions['y'])
    all_positions['z'] = np.concatenate(all_positions['z'])

    visible_positions['x'] = np.concatenate(visible_positions['x'])
    visible_positions['y'] = np.concatenate(visible_positions['y'])
    visible_positions['z'] = np.concatenate(visible_positions['z'])
    
    # Plot the globe with satellite positions
    fig = plot_globe_with_borders()

    # Plot all satellites (red markers)
    fig.add_trace(go.Scatter3d(
        x=all_positions['x'],
        y=all_positions['y'],
        z=all_positions['z'],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.5),
        showlegend=False
    ))

    # Plot visible satellites (green markers)
    fig.add_trace(go.Scatter3d(
        x=visible_positions['x'],
        y=visible_positions['y'],
        z=visible_positions['z'],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.8),
        showlegend=False
    ))

    
    fig.add_trace(go.Scatter3d(
        x=[ground_sta_pos[0] / 1e3],
        y=[ground_sta_pos[1] / 1e3],
        z=[ground_sta_pos[2] / 1e3],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=1.0),
        showlegend=False
    ))

    # Display the plot
    fig.show()

