# Generates a 3D plot of the Earth with the satellites positions
# https://geopandas.org/en/stable/docs/user_guide/io.html
import os
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.satellite.utils.sat_utils import calc_elevation, lla2ecef
from sharc.satellite.ngso.constants import EARTH_RADIUS_KM


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

    # Plot satellite traces from a time interval
    fig = plot_globe_with_borders()
    pos_vec = orbit.get_satellite_positions_time_interval(initial_time_secs=0, interval_secs=5, n_periods=4)
    fig.add_trace(go.Scatter3d(x=pos_vec['sx'].flatten(),
                               y=pos_vec['sy'].flatten(),
                               z=pos_vec['sz'].flatten(),
                               mode='lines',
                               showlegend=False))

    fig.show()

    # Plot satellites positions taken randomly
    fig = plot_globe_with_borders()
    pos_vec = orbit.get_satellite_positions_time_interval()
    NUM_DROPS = 100
    rng = np.random.RandomState(seed=6)
    acc_pos = {'x': list(), 'y': list(), 'z': list()}
    for i in range(NUM_DROPS):
        pos_vec = orbit.get_orbit_positions_random_time(rng=rng)
        acc_pos['x'].extend(pos_vec['sx'].flatten())
        acc_pos['y'].extend(pos_vec['sy'].flatten())
        acc_pos['z'].extend(pos_vec['sz'].flatten())

    fig.add_trace(go.Scatter3d(x=acc_pos['x'],
                               y=acc_pos['y'],
                               z=acc_pos['z'],
                               mode='markers',
                               marker=dict(size=2,
                                           color='red',
                                           opacity=0.8),
                               showlegend=False))
    fig.show()

    # Show visible satellites from ground-station
    GROUND_STA_LAT = -15.7801
    GROUND_STA_LON = -42.9292
    MIN_ELEV_ANGLE_DEG = 30.0
    fig = plot_globe_with_borders()
    pos_vec = orbit.get_satellite_positions_time_interval()
    NUM_DROPS = 100
    rng = np.random.RandomState(seed=6)
    acc_pos = {'x': list(), 'y': list(), 'z': list(), 'lat': list(), 'lon': list()}
    for i in range(NUM_DROPS):
        pos_vec = orbit.get_orbit_positions_random_time(rng=rng)
        acc_pos['x'].extend(pos_vec['sx'].flatten())
        acc_pos['y'].extend(pos_vec['sy'].flatten())
        acc_pos['z'].extend(pos_vec['sz'].flatten())
        acc_pos['lat'].extend(pos_vec['lat'].flatten())
        acc_pos['lon'].extend(pos_vec['lon'].flatten())

    fig.add_trace(go.Scatter3d(x=acc_pos['x'],
                               y=acc_pos['y'],
                               z=acc_pos['z'],
                               mode='markers',
                               marker=dict(size=2,
                                           color='red',
                                           opacity=0.8),
                               showlegend=False))

    lookangles = calc_elevation(GROUND_STA_LAT, acc_pos['lat'], GROUND_STA_LON, acc_pos['lon'], 1414.0)
    vis_sat_idxs = np.where(lookangles > MIN_ELEV_ANGLE_DEG)[0]

    fig.add_trace(go.Scatter3d(
        x=np.array(acc_pos['x'])[vis_sat_idxs],
        y=np.array(acc_pos['y'])[vis_sat_idxs],
        z=np.array(acc_pos['z'])[vis_sat_idxs],
        mode='markers',
        marker=dict(size=2,
                    color='green',
                    opacity=0.8),
        showlegend=False))

    groud_sta_pos = lla2ecef(GROUND_STA_LAT, GROUND_STA_LON, 0.0)
    fig.add_trace(go.Scatter3d(
        x=np.array(groud_sta_pos[0] / 1e3),
        y=np.array(groud_sta_pos[1] / 1e3),
        z=np.array(groud_sta_pos[2] / 1e3),
        mode='markers',
        marker=dict(size=4,
                    color='blue',
                    opacity=1.0),
        showlegend=False))

    fig.show()
