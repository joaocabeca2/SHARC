"""Half-duplex Wi-Fi links (wifi.prob_DL): SystemWifi._run_csma_ca_links gives
each winning AP-STA link a single transmitter (AP with probability prob_DL)
and the other end as receiver; nearby transmitters on the same floor are
blocked as in the original CSMA."""
import unittest
from types import SimpleNamespace

import numpy as np

from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.system.system_wifi import SystemWifi


def _system(n_links, prob_dl, dist):
    ap, sta = StationManager(n_links), StationManager(n_links)
    ap.station_type, sta.station_type = StationType.WIFI_APS, StationType.WIFI_STA
    ap.active = np.ones(n_links, dtype=bool)          # every AP wants to transmit
    sta.active = np.zeros(n_links, dtype=bool)        # STA draws are not used
    s = SimpleNamespace(ap=ap, sta=sta, parameters=SimpleNamespace(prob_DL=prob_dl),
                        link={i: [i] for i in range(n_links)})
    s.draw_csma_links = lambda random_gen: SystemWifi.draw_csma_links(s, random_gen)
    d = np.full((n_links, n_links), dist, dtype=float)
    dist_map = {
        (StationType.WIFI_APS, StationType.WIFI_APS): d,
        (StationType.WIFI_APS, StationType.WIFI_STA): d,
        (StationType.WIFI_STA, StationType.WIFI_APS): d,
        (StationType.WIFI_STA, StationType.WIFI_STA): d,
    }
    return s, dist_map


def _run(n_links, prob_dl, dist=np.inf, seed=0):
    s, dist_map = _system(n_links, prob_dl, dist)
    SystemWifi._run_csma_ca_links(s, np.random.RandomState(seed), dist_map, 20.0)
    return s


class HalfDuplexLinksTest(unittest.TestCase):

    def test_one_transmitter_per_link(self):
        s = _run(200, 0.5)
        # all links win (no neighbour within the radius)
        self.assertTrue(s.ap.active.all() and s.sta.active.all())
        # each link: exactly one transmitter and one receiver, never both
        ap_tx, sta_tx = set(s.ap_tx), set(s.sta_tx)
        for i in range(200):
            self.assertNotEqual(i in ap_tx, i in sta_tx)
        self.assertEqual(set(s.ap_rx), set(range(200)) - ap_tx)
        self.assertEqual(set(s.sta_rx), set(range(200)) - sta_tx)

    def test_prob_dl_extremes_and_mean(self):
        self.assertEqual(len(_run(50, 1.0).sta_tx), 0)
        self.assertEqual(len(_run(50, 0.0).ap_tx), 0)
        frac = np.mean([len(_run(400, 0.5, seed=k).ap_tx) / 400 for k in range(10)])
        self.assertAlmostEqual(frac, 0.5, delta=0.03)

    def test_close_transmitters_block_each_other(self):
        # all transmitters within the radius: only one link wins the channel
        s = _run(30, 0.5, dist=5.0)
        self.assertEqual(s.ap.active.sum(), 1)
        self.assertEqual(len(s.ap_tx) + len(s.sta_tx), 1)


if __name__ == "__main__":
    unittest.main()
