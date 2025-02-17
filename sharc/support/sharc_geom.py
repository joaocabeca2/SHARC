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

def get_rotation_matrix(around_z, around_y):
    """
    Rotates with the right hand rule around the z axis (similar to simulator azimuth)
    and with the right hand rule around the y axis (similar to simulator elevation)
    """
    alpha = np.deg2rad(around_z)
    beta = np.deg2rad(around_y)

    ry = np.matrix([
        [np.cos(beta), 0.0, np.sin(beta)],
        [0.0, 1.0, 0.0],
        [-np.sin(beta), 0.0, np.cos(beta)],
    ])
    rz = np.matrix([
        [np.cos(alpha), -np.sin(alpha), 0.0],
        [np.sin(alpha), np.cos(alpha), 0.0],
        [0.0, 0.0, 1.0],
    ])

    return rz * ry

def rotate_angles_based_on_new_nadir(elev, azim, nadir_elev, nadir_azim):
    """
    Receives elevation and azimuth 2d, rotates around base
    so that base_elev <> 0deg and base_azim <> 0deg
    elevation being 0 at horizon (xy plane) and azimuth 0 at x axis

    Returns
    ------
        elevation, azimuth
        (xy plane elevation)
    """
    # translating to normal polar coordinate system, with theta being angle from z axis
    # and phi being angle from x axis in the xy plane
    nadir_theta = 90 - nadir_elev
    nadir_phi = nadir_azim

    nadir_point = np.matrix([
        np.sin(np.deg2rad(nadir_theta)) * np.cos(np.deg2rad(nadir_phi)),
        np.sin(np.deg2rad(nadir_theta)) * np.sin(np.deg2rad(nadir_phi)),
        np.cos(np.deg2rad(nadir_theta)),
    ])
    # first rotate around y axis nadir_theta-180 to reach new theta
    # since nadir_theta in (0,180), rotation will end up to azimuth=0
    # so we rotate it around z axis nadir_phi
    rotation_matrix = get_rotation_matrix(nadir_phi, nadir_theta-180)

    theta = 90 - elev
    phi = azim

    phi_rad = np.ravel(np.array([np.deg2rad(phi)]))
    theta_rad = np.ravel(np.array([np.deg2rad(theta)]))

    points = np.matrix([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad),
    ])

    rotated_points = rotation_matrix @ points

    rotated_phi = np.ravel(
        np.asarray(
            np.rad2deg(
                np.arctan2(rotated_points[1], rotated_points[0]),
            ),
        ),
    )
    rotated_theta = np.ravel(
        np.asarray(
            np.rad2deg(np.arccos(rotated_points[2])),
        ),
    )

    # back to elevation = 0 deg at xy plane, 90 at zenith and -90 at nadir
    res_elev = 90 - rotated_theta

    return res_elev, rotated_phi

if __name__ == "__main__":
    # baixo, direita, frente, esquerda, atrás, cima, cima
    # elev = np.array([-90., 0., 0., 0., 0., 90., 90.])
    # azim = np.array([0., 0., 90., 180., -90., 0., 90.])
    elev = np.array([-89.93761622])
    azim = np.array([180.])
    n = len(azim)

    print("pointing nadir to the right (rotated around earth to the left)")
    print(np.concatenate((elev, azim)).reshape((2, n)).transpose())
    res_elev, res_az = rotate_angles_based_on_new_nadir(elev, azim, 0, 0)
    res = np.concatenate((res_elev, res_az)).reshape((2, n)).transpose()
    print(res)

    print("pointing nadir to the left (rotated around earth to the right)")
    print(np.concatenate((elev, azim)).reshape((2, n)).transpose())
    res_elev, res_az = rotate_angles_based_on_new_nadir(elev, azim, 0, 180)
    res = np.concatenate((res_elev, res_az)).reshape((2, n)).transpose()
    print(res)
