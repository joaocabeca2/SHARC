"""Tests for the Wi-Fi CSMA/CA with CCA by received power (wifi.cca_mode POWER)."""
import unittest

import numpy as np

from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.system.system_wifi import SystemWifi

BW = 80.0
TO_20 = 10 * np.log10(20.0 / BW)      # -6.02 dB: power over 80 MHz -> per 20 MHz
PD, ED = -82.0, -62.0
NO_IMT = -np.inf


def contention(rx, imt):
    rx = np.array(rx, dtype=float)
    return SystemWifi.cca_power_contention(rx, np.array(imt, dtype=float), BW, PD, ED)


class TestCcaPowerContention(unittest.TestCase):

    def test_first_link_always_transmits_without_imt(self):
        rx = [[-np.inf, -40.0], [-40.0, -np.inf]]
        tx, imt = contention(rx, [NO_IMT, NO_IMT])
        np.testing.assert_array_equal(tx, [True, False])
        np.testing.assert_array_equal(imt, [False, False])

    def test_preamble_detection_threshold_per_20_mhz(self):
        # just above / just below -82 dBm per 20 MHz (power over 80 MHz is 6 dB higher)
        above = PD - TO_20 + 0.1
        below = PD - TO_20 - 0.1
        tx, _ = contention([[-np.inf, above], [above, -np.inf]], [NO_IMT, NO_IMT])
        np.testing.assert_array_equal(tx, [True, False])
        tx, _ = contention([[-np.inf, below], [below, -np.inf]], [NO_IMT, NO_IMT])
        np.testing.assert_array_equal(tx, [True, True])

    def test_energy_detection_sums_wifi_transmitters(self):
        # link 2 receives two winners each below PD, whose sum is below ED -> transmits;
        # with IMT energy added, the total exceeds ED -> defers
        p = PD - TO_20 - 1.0                      # each winner below PD
        rx = np.full((3, 3), -np.inf)
        rx[0, 2] = rx[1, 2] = p
        tx, _ = contention(rx, [NO_IMT, NO_IMT, NO_IMT])
        np.testing.assert_array_equal(tx, [True, True, True])
        imt_just_below_ed = ED - TO_20 - 0.01     # IMT alone does not block
        tx, imt = contention(rx, [NO_IMT, NO_IMT, imt_just_below_ed])
        np.testing.assert_array_equal(tx, [True, True, False])
        np.testing.assert_array_equal(imt, [False, False, False])

    def test_imt_blocks_regardless_of_order(self):
        rx = np.full((3, 3), -np.inf)
        imt = [ED - TO_20 + 1.0, NO_IMT, NO_IMT]  # first in the order, but above ED
        tx, blocked_imt = contention(rx, imt)
        np.testing.assert_array_equal(tx, [False, True, True])
        np.testing.assert_array_equal(blocked_imt, [True, False, False])

    def test_blocked_link_does_not_block_others(self):
        # link 1 is blocked by link 0; link 2 hears only link 1 (strongly) -> transmits
        rx = np.full((3, 3), -np.inf)
        rx[0, 1] = -40.0
        rx[1, 2] = -40.0
        tx, _ = contention(rx, [NO_IMT] * 3)
        np.testing.assert_array_equal(tx, [True, False, True])

    def test_empty(self):
        tx, imt = contention(np.zeros((0, 0)), [])
        self.assertEqual(tx.size, 0)
        self.assertEqual(imt.size, 0)


class TestCcaParameters(unittest.TestCase):

    def _params(self, **kw):
        p = ParametersWifiSystem()
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def test_defaults(self):
        p = ParametersWifiSystem()
        self.assertEqual(p.cca_mode, "DISTANCE")
        self.assertEqual(p.cca_pd_threshold, -82.0)
        self.assertEqual(p.cca_ed_threshold, -62.0)
        self.assertTrue(p.cca_detect_imt)


if __name__ == "__main__":
    unittest.main()
