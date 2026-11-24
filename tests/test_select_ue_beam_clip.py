"""Beam steering limits in Simulation.select_ue, as in ITU-R WP 5D (Annex 4.2
to Doc. 5D/413): horizontal range relative to the sector azimuth (the clip must
not break across +-180 deg); vertical range in the global system (90 deg =
horizon), already including the mechanical downtilt."""
import unittest
from types import SimpleNamespace

import numpy as np

from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.simulation import Simulation
from sharc.station_manager import StationManager


class _BeamRecorder:
    """Records the beams (global angles) with the real antenna rotation."""

    to_local_coord = AntennaBeamformingImt.to_local_coord
    to_global_coord = AntennaBeamformingImt.to_global_coord

    def __init__(self, azimuth=0.0, elevation=0.0):
        self.azimuth = azimuth
        self.elevation = elevation
        AntennaBeamformingImt._calculate_rotation_matrix(self)
        self.beams = []
        self.beams_list = self.beams

    def add_beam(self, phi, theta):
        self.beams.append((float(np.ravel(phi)[0]), float(np.ravel(theta)[0])))


def _run_select_ue(bs_azimuth, ue_phi_deg, ue_dist=100.0, downtilt=0.0,
                   vertical_range=(0.0, 180.0)):
    """One BS (18 m) with the given sector azimuth and one UE (1.5 m) at
    ue_phi_deg and ue_dist. Returns the beam (phi, theta) in global angles."""
    bs = StationManager(1)
    bs.x, bs.y, bs.z, bs.height = np.zeros(1), np.zeros(1), np.full(1, 18.0), np.full(1, 18.0)
    bs.azimuth = np.array([bs_azimuth], dtype=float)
    bs.active = np.ones(1, dtype=bool)
    bs.antenna = [_BeamRecorder(bs_azimuth, -downtilt)]

    ue = StationManager(1)
    phi = np.radians(ue_phi_deg)
    ue.x, ue.y = np.array([ue_dist * np.cos(phi)]), np.array([ue_dist * np.sin(phi)])
    ue.z, ue.height = np.full(1, 1.5), np.full(1, 1.5)
    ue.active = np.zeros(1, dtype=bool)
    ue.antenna = [_BeamRecorder()]

    array = SimpleNamespace(
        horizontal_beamsteering_range=np.array([-60.0, 60.0]),
        vertical_beamsteering_range=np.array(vertical_range),
    )
    sim = SimpleNamespace(
        wrap_around_enabled=False,
        bs=bs,
        ue=ue,
        link={0: [0]},
        bs_to_ue_beam_rbs=-np.ones(1, dtype=int),
        parameters=SimpleNamespace(imt=SimpleNamespace(
            ue=SimpleNamespace(k=1),
            bs=SimpleNamespace(antenna=SimpleNamespace(array=array)),
        )),
    )
    Simulation.select_ue(sim, np.random.RandomState(0))
    return bs.antenna[0].beams[0]


def _depression(ue_dist):
    """Global theta of the UE seen from the BS (18 m -> 1.5 m)."""
    return 90 + np.degrees(np.arctan((18.0 - 1.5) / ue_dist))


class SelectUeBeamClipTest(unittest.TestCase):

    def assertSameAngle(self, a, b, places=6):
        self.assertAlmostEqual(((a - b + 180) % 360) - 180, 0.0, places=places)

    def test_ue_inside_sector_is_not_clipped(self):
        for az, ue_phi in [(-180, 170), (-180, -170), (180, 125), (60, 100),
                           (-60, -110), (-180, 121), (-180, -121)]:
            with self.subTest(az=az, ue_phi=ue_phi):
                self.assertSameAngle(_run_select_ue(az, ue_phi)[0], ue_phi)

    def test_ue_outside_range_is_clipped_to_nearest_edge(self):
        # sector -180: edges at +120 and -120
        self.assertSameAngle(_run_select_ue(-180, 100)[0], 120)
        self.assertSameAngle(_run_select_ue(-180, -100)[0], -120)
        self.assertSameAngle(_run_select_ue(60, -30)[0], 0)

    def test_vertical_range_is_global_including_downtilt(self):
        # ITU-R WP 5D (Annex 4.2 to Doc. 5D/413), Note 1: the vertical
        # coverage range is in the global system (90 deg = horizon) and
        # already includes the mechanical downtilt. With [90, 100] the beam
        # covers horizon to 10 deg below it, whatever the downtilt.
        for downtilt in (6.0, 10.0):
            kw = dict(downtilt=downtilt, vertical_range=(90.0, 100.0))
            with self.subTest(downtilt=downtilt):
                # far UE (about 3 deg below horizon): inside, beam at the UE
                phi, theta = _run_select_ue(-180, 180, ue_dist=300.0, **kw)
                self.assertAlmostEqual(theta, _depression(300.0), 6)
                self.assertSameAngle(phi, 180)
                # UE 15 deg below horizon: clipped at 10 deg down
                d = 16.5 / np.tan(np.radians(15.0))
                self.assertAlmostEqual(_run_select_ue(60, 60, ue_dist=d, **kw)[1], 100.0, 6)

    def test_horizontal_clip_with_downtilted_panel(self):
        # the horizontal range stays relative to the sector azimuth
        kw = dict(downtilt=10.0, vertical_range=(90.0, 100.0))
        self.assertSameAngle(_run_select_ue(-180, 100, **kw)[0], 120)
        self.assertSameAngle(_run_select_ue(60, 100, **kw)[0], 100)

    def test_local_global_roundtrip(self):
        ant = _BeamRecorder(azimuth=-180.0, elevation=-10.0)
        phi = np.array([-179.0, 175.0, 120.5, -60.0])
        theta = np.array([93.0, 101.0, 115.0, 89.0])
        g_phi, g_theta = ant.to_global_coord(*ant.to_local_coord(phi, theta))
        np.testing.assert_allclose(((g_phi - phi + 180) % 360) - 180, 0, atol=1e-9)
        np.testing.assert_allclose(g_theta, theta, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
