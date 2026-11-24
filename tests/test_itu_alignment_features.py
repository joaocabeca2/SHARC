"""Tests for the features added to align the IMT x Wi-Fi model with ITU-R
WP 5D (Annex 4.2 to Doc. 5D/413) and the improvement plan of 2026-11-06:

- vectorized sub-array gain (same values as the scalar computation);
- Wi-Fi AP beamforming antenna (normalization to 0 dBi mean gain, peak gain,
  single beam pointed at the STA);
- e.i.r.p. limit of the Wi-Fi nodes;
- building entry loss (P.2109) on the IMT links of indoor UEs.
"""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.antenna.antenna_beamforming_wifi import AntennaBeamformingWifi
from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.propagation.propagation_building_entry_loss import PropagationBuildingEntryLoss
from sharc.simulation import Simulation
from sharc.station_manager import StationManager


def _itu_bs_params():
    """BS antenna of ITU-R WP 5D, macro urban, 7125-8400 MHz."""
    p = ParametersAntennaImt()
    p.n_rows, p.n_columns = 16, 8
    p.element_pattern, p.element_max_g = "M2101", 6.4
    p.element_phi_3db, p.element_theta_3db = 90, 65
    p.element_am = p.element_sla_v = 30
    p.element_horiz_spacing, p.element_vert_spacing = 0.5, 2.1
    p.normalization, p.minimum_array_gain = False, -200
    p.adjacent_antenna_model = "BEAMFORMING"
    p.subarray.is_enabled = True
    p.subarray.n_rows, p.subarray.element_vert_spacing, p.subarray.eletrical_downtilt = 3, 0.7, 3.0
    return p


def _wifi_params(rows, cols, pattern="FIXED", gmax=0.0, th3=90.0):
    p = ParametersAntennaWifi()
    p.beamforming = True
    p.n_rows, p.n_columns = rows, cols
    p.element_pattern, p.element_max_g = pattern, gmax
    p.element_phi_3db, p.element_theta_3db = 90.0, th3
    p.adjacent_antenna_model = "BEAMFORMING"
    return p


def _sphere():
    th = np.arange(0.5, 180.0, 1.0)
    ph = np.arange(-179.5, 180.0, 1.0)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return PH.ravel(), TH.ravel(), np.sin(np.radians(TH.ravel()))


class SubarrayVectorizedTest(unittest.TestCase):

    def test_vectorized_equals_scalar(self):
        a = AntennaBeamformingImt(_itu_bs_params(), 30.0, -6.0)
        a.add_beam(40.0, 95.0)
        a.add_beam(10.0, 99.0)
        rng = np.random.RandomState(1)
        ph, th = rng.uniform(-180, 180, 500), rng.uniform(0, 180, 500)
        beams = rng.randint(-1, 2, 500)
        lp, lt = a.to_local_coord(ph, th)
        ref = np.array([a._beam_gain(x, y, b) for x, y, b in zip(lp, lt, beams)])
        np.testing.assert_allclose(a._beam_gain_vec(lp, lt, beams), ref, atol=1e-9)

    def test_itu_eirp(self):
        # 22 dBm per sub-array and polarization, 16 x 8 x 2, beam at 9 deg
        # below the horizon (6 deg mechanical + 3 deg sub-array tilt):
        # e.i.r.p. per sector = 78.3 dBm (Annex 4.2, row 1.13)
        a = AntennaBeamformingImt(_itu_bs_params(), 0.0, -6.0)
        a.add_beam(0.0, 99.0)
        g = a.calculate_gain(phi_vec=np.array([0.0]), theta_vec=np.array([99.0]), beams_l=np.array([0]))[0]
        self.assertAlmostEqual(22 + 10 * np.log10(2 * 16 * 8) + g, 78.3, places=1)


class WifiApBeamformingTest(unittest.TestCase):

    def _mean_gain_db(self, ant, phi_b, theta_b):
        ant.reset_beams()
        ant.add_beam(phi_b, theta_b)
        ph, th, w = _sphere()
        g = ant.calculate_gain(phi_vec=ph, theta_vec=th, beams_l=np.zeros(ph.size, int))
        return 10 * np.log10(np.sum(w * 10 ** (g / 10)) / np.sum(w))

    def test_normalized_mean_gain_is_0_dbi(self):
        for par, el in ((_wifi_params(1, 4), 0.0),
                        (_wifi_params(2, 2, "M2101", 5.0, 150.0), -90.0)):
            with self.subTest(rows=par.n_rows, cols=par.n_columns):
                ant = AntennaBeamformingWifi(par, 0.0, el)
                self.assertAlmostEqual(self._mean_gain_db(ant, 0.0, 90.0 - el), 0.0, delta=0.05)

    def test_linear_4x1_isotropic_peak_gain(self):
        ant = AntennaBeamformingWifi(_wifi_params(1, 4), 0.0, 0.0)
        self.assertAlmostEqual(ant.norm_offset_db, 0.0, delta=0.05)
        self.assertAlmostEqual(ant.peak_gain, 10 * np.log10(4), delta=0.05)

    def test_beam_points_at_sta(self):
        ant = AntennaBeamformingWifi(_wifi_params(1, 4), 30.0, 0.0)
        ant.add_beam(50.0, 95.0)
        g_sta = ant.calculate_gain(phi_vec=np.array([50.0]), theta_vec=np.array([95.0]),
                                   beams_l=np.array([0]))[0]
        self.assertAlmostEqual(g_sta, ant.peak_gain, delta=0.3)
        # along the array axis (end-fire, local phi = 90) the gain is far lower
        g_axis = ant.calculate_gain(phi_vec=np.array([120.0]), theta_vec=np.array([90.0]),
                                    beams_l=np.array([0]))[0]
        self.assertLess(g_axis, g_sta - 10)


class WifiEirpLimitTest(unittest.TestCase):

    def test_conducted_power_with_eirp_limit(self):
        f = Simulation.wifi_conducted_power
        self.assertEqual(f(23.0, 6.0, 3.0, None), 23.0)       # no limit
        self.assertEqual(f(23.0, 0.0, 3.0, 24.0), 23.0)       # 20 dBm < 24
        self.assertAlmostEqual(f(23.0, 6.02, 3.0, 24.0), 20.98)  # 26.02 -> 24
        self.assertAlmostEqual(f(23.0, 0.0, 3.0, 18.0), 21.0)    # STA 20 -> 18


class IndoorUeBelTest(unittest.TestCase):

    def _coupling(self, bel_on):
        ue = StationManager(4)
        ue.indoor = np.array([True, False, True, False])
        bs = StationManager(2)
        sim = SimpleNamespace(
            parameters=SimpleNamespace(imt=SimpleNamespace(
                frequency=7000.0,
                bs=SimpleNamespace(ohmic_loss=0.0),
                ue=SimpleNamespace(ohmic_loss=0.0, body_loss=0.0,
                                   indoor_bel_building_class="TRADITIONAL"))),
            calculate_gains=lambda a, b: np.zeros((a.num_stations, b.num_stations)),
            propagation_imt=SimpleNamespace(
                get_loss=lambda *args: np.full((4, 2), 100.0)),
            bs_to_ue_theta=np.full((2, 4), 95.0),
            imt_indoor_bel=PropagationBuildingEntryLoss(np.random.RandomState(3)) if bel_on else None,
        )
        return Simulation.calculate_intra_imt_coupling_loss(sim, ue, bs)

    def test_bel_only_on_indoor_ues(self):
        np.testing.assert_allclose(self._coupling(False), 100.0)
        cl = self._coupling(True)
        np.testing.assert_allclose(cl[:, [1, 3]], 100.0)          # outdoor: unchanged
        self.assertTrue(np.all(cl[:, [0, 2]] > 100.0))            # indoor: + BEL
        # independent draws per BS-UE pair
        self.assertNotAlmostEqual(cl[0, 0], cl[1, 0])


if __name__ == "__main__":
    unittest.main()
