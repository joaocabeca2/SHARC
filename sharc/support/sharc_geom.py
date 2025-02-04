import numpy as np

def cartesian_to_polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Converts cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : np.ndarray
        x coordinate in meters
    y : np.ndarray
        y coordinate in meters
    z : np.ndarray
        z coordinate in meters

    Returns
    -------
    tuple
        range, azimuth and elevation in meters, degrees and degrees
    """
    # range calculation
    r = np.sqrt(x**2 + y**2 + z**2)

    # azimuth calculation
    azimuth = np.arctan2(y, x)

    # elevation calculation
    elevation = np.arcsin(z / r)

    return r, np.degrees(azimuth), np.degrees(elevation)


def polar_to_cartesian(r: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray) -> tuple:
    """
    Converts polar coordinates to cartesian coordinates.

    Parameters
    ----------
    r : np.ndarray
        range in meters
    azimuth : np.ndarray
        azimuth in degrees
    elevation : np.ndarray
        elevation in degrees

    Returns
    -------
    tuple
        x, y and z coordinates in meters
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return x, y, z

