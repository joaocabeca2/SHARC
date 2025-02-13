# Generates a 3D plot of the Earth with the satellites positions
# https://geopandas.org/en/stable/docs/user_guide/io.html

import os
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance

from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.satellite.utils.sat_utils import calc_elevation, lla2ecef
from sharc.satellite.ngso.constants import EARTH_RADIUS_KM
from sharc.station_factory import StationFactory
from sharc.parameters.imt.parameters_imt import ParametersImt
from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
from sharc.topology.topology_single_base_station_spherical import TopologySingleBaseStationSpherical


# Defined position for IMT BS
GROUND_STA_LAT = -15.7801
GROUND_STA_LON = -42.9292
CELL_RADIUS = 1000
NUM_CLUSTERS = 1

# Elevation angle for the lookangles calculation
MIN_ELEV_ANGLE_DEG = 30.0

# Parameters of IMT earth system
param_imt = ParametersImt()
param_imt.ue.distribution_type = "UNIFORM_IN_CELL"

# Parameters of IMT antenna system
ue_param_ant = ParametersAntennaImt()

# Seed generation
seed = 100
rng = np.random.RandomState(seed)

# Topology parameters ()
topology = TopologySingleBaseStationSpherical(

    cell_radius = CELL_RADIUS,
    num_clusters = NUM_CLUSTERS,
    central_latitude = GROUND_STA_LAT,
    central_longitude = GROUND_STA_LON

)

topology.type = "OUTDOOR"

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

    # Plot Global Star orbit using OrbitModel object
    orbit = OrbitModel(
        Nsp=6,
        Np=8,
        phasing=7.5,
        long_asc=0,
        omega=0,
        delta=52,
        hp=1414,
        ha=1414,
        Mo=0
    )

    fig = plot_globe_with_borders()

    # Pushing IMT UE positions from station_factory 
    topology.calculate_coordinates()
    imt_ue_positions = StationFactory.generate_imt_ue_outdoor(param_imt, ue_param_ant, rng, topology)   
    
    for i in range(len(imt_ue_positions.x)):
        fig.add_trace(go.Scatter3d(
            x=np.array(imt_ue_positions.x[i]/1e3),
            y=np.array(imt_ue_positions.y[i]/1e3),
            z=np.array(imt_ue_positions.z[i]/1e3),
            mode='markers',
            marker=dict(size=3,
                    color='blue',
                    opacity=1.0),
            showlegend=True))
    
    # Generating the plot for the reference BS
    ground_sta_pos = lla2ecef(GROUND_STA_LAT, GROUND_STA_LON, 0.0)
    fig.add_trace(go.Scatter3d(
        x=np.array(ground_sta_pos[0] / 1e3),
        y=np.array(ground_sta_pos[1] / 1e3),
        z=np.array(ground_sta_pos[2] / 1e3),
        mode='markers',
        marker=dict(size=4,
                    color='red',
                    opacity=1.0),
        showlegend=True))
  
    fig.show()
