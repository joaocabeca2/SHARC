import unittest
import os
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.satellite.utils.sat_utils import calc_elevation, lla2ecef
from sharc.satellite.ngso.constants import EARTH_RADIUS_KM
from sharc.station_factory import StationFactory
from sharc.parameters.imt.parameters_imt import ParametersImt
from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
from sharc.topology.topology_single_base_station_spherical import TopologySingleBaseStationSpherical

class SimulateUETest(unittest.TestCase):
    
    # Defined position for IMT Base Station
    GROUND_STA_LAT = -15.7801
    GROUND_STA_LON = -42.9292
    CELL_RADIUS = 1000
    NUM_CLUSTERS = 1

    # Minimum elevation angle for look-angle calculation
    MIN_ELEV_ANGLE_DEG = 30.0

    # IMT earth system parameters
    param_imt = ParametersImt()
    param_imt.ue.distribution_type = "UNIFORM_IN_CELL"

    # IMT antenna system parameters
    ue_param_ant = ParametersAntennaImt()

    # Seed for random generation
    seed = 100
    rng = np.random.RandomState(seed)

    # Topology parameters
    topology = TopologySingleBaseStationSpherical(
        cell_radius=CELL_RADIUS,
        num_clusters=NUM_CLUSTERS,
        central_latitude=GROUND_STA_LAT,
        central_longitude=GROUND_STA_LON
    )
    topology.type = "OUTDOOR"

    def plot_half_sphere(self, fig, is_front=True):
        """Plots half of the sphere (Earth)."""
        color = 'rgb(220, 220, 220)'
        R = np.sqrt(EARTH_RADIUS_KM)
        u_angle = np.linspace(-np.pi if is_front else 0, 0 if is_front else np.pi, 25)
        v_angle = np.linspace(0, np.pi, 25)
        x_dir = np.outer(R * np.cos(u_angle), R * np.sin(v_angle))
        y_dir = np.outer(R * np.sin(u_angle), R * np.sin(v_angle))
        z_dir = np.outer(R * np.ones(u_angle.shape[0]), R * np.cos(v_angle))
        fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, color], [1, color]],
                        opacity=1.0, showlegend=False, lighting=dict(diffuse=0.1))

    def plot_polygon(self, poly):
        """Extracts and transforms polygon coordinates for plotting."""
        xy_coords = poly.exterior.coords.xy
        lon = np.radians(np.array(xy_coords[0]))
        lat = np.radians(np.array(xy_coords[1]))

        R = EARTH_RADIUS_KM
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)

        return x, y, z

    def plot_globe_with_borders(self):
        """Creates a 3D plot of the Earth with country borders."""
        countries_borders_shp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "../data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
        gdf = gpd.read_file(countries_borders_shp_file)
        fig = go.Figure()
        self.plot_half_sphere(fig, is_front=True)
        self.plot_half_sphere(fig, is_front=False)

        for _, row in gdf.iterrows():
            polys = row.geometry
            if polys.geom_type == 'Polygon':
                x, y, z = self.plot_polygon(polys)
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                           line=dict(color='rgb(0, 0, 0)'), showlegend=False))
            elif polys.geom_type == 'MultiPolygon':
                for poly in polys.geoms:
                    x, y, z = self.plot_polygon(poly)
                    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                               line=dict(color='rgb(0, 0, 0)'), showlegend=False))
        return fig

    def test_plot_satellite_positions(self):
        """Tests the plotting of satellite and UE positions on a 3D globe."""
        orbit = OrbitModel(
            Nsp=6, Np=8, phasing=7.5, long_asc=0,
            omega=0, delta=52, hp=1414, ha=1414, Mo=0
        )
        
        fig = self.plot_globe_with_borders()
        self.topology.calculate_coordinates()
        imt_ue_positions = StationFactory.generate_imt_ue_outdoor(self.param_imt, self.ue_param_ant, self.rng, self.topology)

        for i in range(len(imt_ue_positions.x)):
            fig.add_trace(go.Scatter3d(
                x=np.array(imt_ue_positions.x[i]/1e3),
                y=np.array(imt_ue_positions.y[i]/1e3),
                z=np.array(imt_ue_positions.z[i]/1e3),
                mode='markers',
                marker=dict(size=3, color='blue', opacity=1.0),
                showlegend=True))

        ground_sta_pos = lla2ecef(self.GROUND_STA_LAT, self.GROUND_STA_LON, 0.0)
        fig.add_trace(go.Scatter3d(
            x=np.array(ground_sta_pos[0] / 1e3),
            y=np.array(ground_sta_pos[1] / 1e3),
            z=np.array(ground_sta_pos[2] / 1e3),
            mode='markers',
            marker=dict(size=4, color='red', opacity=1.0),
            showlegend=True))

        fig.show()

if __name__ == "__main__":
    unittest.main()
