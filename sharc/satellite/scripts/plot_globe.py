import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
from pathlib import Path

from sharc.satellite.ngso.constants import EARTH_RADIUS_KM


def plot_back(fig, geoconv, in_km):
    """back half of sphere"""
    clor = 'rgb(220, 220, 220)'
    # Create a mesh grid for latitude and longitude.
    # For the "front" half, we can use longitudes from -180 to 0 degrees.
    lat_vals = np.linspace(-90, 90, 50)       # 50 latitude points from -90 to 90 degrees.
    lon_vals = np.linspace(0, 180, 50)         # 50 longitude points for the front half.
    lon, lat = np.meshgrid(lon_vals, lat_vals)  # lon and lat will be 2D arrays.

    # Flatten the mesh to pass to the converter function.
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()

    if geoconv is None:
        lon = lon * np.pi / 180
        lat = lat * np.pi / 180

        R = EARTH_RADIUS_KM
        if not in_km:
            R *= 1e3
        x_flat = R * np.cos(lat) * np.cos(lon)
        y_flat = R * np.cos(lat) * np.sin(lon)
        z_flat = R * np.sin(lat)
    else:
        # Convert the lat/lon grid to transformed Cartesian coordinates.
        # Ensure your converter function can handle vectorized (numpy array) inputs.
        x_flat, y_flat, z_flat = geoconv.convert_lla_to_transformed_cartesian(lat_flat, lon_flat, 0)
        if in_km:
            x_flat, y_flat, z_flat = x_flat/1e3, y_flat/1e3, z_flat/1e3

    # Reshape the converted coordinates back to the 2D grid shape.
    x = x_flat.reshape(lat.shape)
    y = y_flat.reshape(lat.shape)
    z = z_flat.reshape(lat.shape)

    # Add the surface to the Plotly figure.
    fig.add_surface(
        x=x, y=y, z=z,
        colorscale=[[0, clor], [1, clor]],  # Uniform color scale for a solid color.
        opacity=1.0,
        showlegend=False,
        lighting=dict(diffuse=0.1),
        showscale=False
    )


def plot_front(fig, geoconv, in_km):
    """front half of sphere"""
    clor = 'rgb(220, 220, 220)'

    # Create a mesh grid for latitude and longitude.
    # For the "front" half, we can use longitudes from -180 to 0 degrees.
    lat_vals = np.linspace(-90, 90, 50)       # 50 latitude points from -90 to 90 degrees.
    lon_vals = np.linspace(-180, 0, 50)         # 50 longitude points for the front half.
    lon, lat = np.meshgrid(lon_vals, lat_vals)  # lon and lat will be 2D arrays.

    # Flatten the mesh to pass to the converter function.
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()

    if geoconv is None:
        lon = lon * np.pi / 180
        lat = lat * np.pi / 180

        R = EARTH_RADIUS_KM
        if not in_km:
            R *= 1e3
        x_flat = R * np.cos(lat) * np.cos(lon)
        y_flat = R * np.cos(lat) * np.sin(lon)
        z_flat = R * np.sin(lat)
    else:
        # Convert the lat/lon grid to transformed Cartesian coordinates.
        # Ensure your converter function can handle vectorized (numpy array) inputs.
        x_flat, y_flat, z_flat = geoconv.convert_lla_to_transformed_cartesian(lat_flat, lon_flat, 0)
        if in_km:
            x_flat, y_flat, z_flat = x_flat/1e3, y_flat/1e3, z_flat/1e3

    # Reshape the converted coordinates back to the 2D grid shape.
    x = x_flat.reshape(lat.shape)
    y = y_flat.reshape(lat.shape)
    z = z_flat.reshape(lat.shape)

    # Add the surface to the Plotly figure.
    fig.add_surface(
        x=x, y=y, z=z,
        colorscale=[[0, clor], [1, clor]],  # Uniform color scale for a solid color.
        opacity=1.0,
        showlegend=False,
        lighting=dict(diffuse=0.1),
        showscale=False
    )


def plot_polygon(poly, geoconv, in_km, alt=0):

    xy_coords = poly.exterior.coords.xy
    lon = np.array(xy_coords[0])
    lat = np.array(xy_coords[1])

    if geoconv is None:
        lon = lon * np.pi / 180
        lat = lat * np.pi / 180

        R = EARTH_RADIUS_KM + alt / (1 if in_km else 1e3)
        if not in_km:
            R *= 1e3
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)
    else:
        x, y, z = geoconv.convert_lla_to_transformed_cartesian(lat, lon, alt * (1e3 if in_km else 1))
        if in_km:
            x, y, z = x/1e3, y/1e3, z/1e3

    return x, y, z


def plot_mult_polygon(mult_poly, geoconv, in_km: bool, alt=0):
    if mult_poly.geom_type == 'Polygon':
        return [plot_polygon(mult_poly, geoconv, in_km, alt)]
    elif mult_poly.geom_type == 'MultiPolygon':
        return [plot_polygon(poly, geoconv, in_km, alt) for poly in mult_poly.geoms]


def plot_globe_with_borders(opaque_globe: bool, geoconv, in_km: bool):
    # Read the shapefile.  Creates a DataFrame object
    project_root = Path(__file__).resolve().parents[3]
    countries_borders_shp_file = project_root / "sharc/data/countries/ne_110m_admin_0_countries.shp"
    gdf = gpd.read_file(countries_borders_shp_file)
    fig = go.Figure()
    # fig.update_layout(
    #     scene=dict(
    #         aspectmode="data",
    #         xaxis=dict(showbackground=False),
    #         yaxis=dict(showbackground=False),
    #         zaxis=dict(showbackground=False)
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=0)
    # )
    if opaque_globe:
        plot_front(fig, geoconv, in_km)
        plot_back(fig, geoconv, in_km)
    x_all, y_all, z_all = [], [], []

    for i in gdf.index:
        # print(gdf.loc[i].NAME)            # Call a specific attribute

        polys = gdf.loc[i].geometry         # Polygons or MultiPolygons

        if polys.geom_type == 'Polygon':
            x, y, z = plot_polygon(polys, geoconv, in_km, 1 if in_km else 1e3)
            x_all.extend(x)
            x_all.extend([None])  # None separates different polygons
            y_all.extend(y)
            y_all.extend([None])
            z_all.extend(z)
            z_all.extend([None])

        elif polys.geom_type == 'MultiPolygon':

            for poly in polys.geoms:
                x, y, z = plot_polygon(poly, geoconv, in_km, 1 if in_km else 1e3)
                x_all.extend(x)
                x_all.extend([None])  # None separates different polygons
                y_all.extend(y)
                y_all.extend([None])
                z_all.extend(z)
                z_all.extend([None])

    fig.add_trace(go.Scatter3d(x=x_all, y=y_all, z=z_all, mode='lines',
                               line=dict(color='rgb(0, 0, 0)'), showlegend=False))

    return fig


