"""CSMA/CA contention in SystemWifi: with INDOOR_BUILDING only nodes in the
same building and on the same floor (within max_dist_nodes_wifi) block each
other."""
import unittest
from types import SimpleNamespace

import numpy as np

from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.system.system_wifi import SystemWifi


def _stations(kind, xyz, building, floor, active):
    st = StationManager(len(xyz))
    st.station_type = kind
    xyz = np.asarray(xyz, dtype=float)
    st.x, st.y, st.z = xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy()
    st.height = st.z.copy()
    st.building_id = np.asarray(building)
    st.floor = np.asarray(floor)
    st.active = np.asarray(active, dtype=bool)
    return st


def _run(topology_type, seed):
    # AP0 and AP2: same building and floor, 5 m apart -> only one wins
    # AP1: right above AP0 (next floor) -> not blocked by AP0
    # AP3: other building, 8 m from AP2 -> not blocked by AP2
    ap = _stations(StationType.WIFI_APS,
                   [(0, 0, 1.5), (0, 0, 4.5), (5, 0, 1.5), (13, 0, 1.5)],
                   [0, 0, 0, 1], [0, 1, 0, 0], [True] * 4)
    sta = _stations(StationType.WIFI_STA,
                    [(1, 0, 1.5), (1, 0, 4.5), (6, 0, 1.5), (14, 0, 1.5)],
                    [0, 0, 0, 1], [0, 1, 0, 0], [False] * 4)
    wifi = SimpleNamespace(
        ap=ap, sta=sta, link={i: [i] for i in range(4)},
        parameters=SimpleNamespace(
            topology=SimpleNamespace(type=topology_type),
            max_dist_nodes_wifi=20.0,
        ),
    )
    SystemWifi.run_csma_ca_scheduling(wifi, np.random.RandomState(seed))
    return ap.active


class WifiCsmaFloorsTest(unittest.TestCase):

    def test_only_same_building_and_floor_contend(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                active = _run("INDOOR_BUILDING", seed)
                self.assertTrue(active[1])                   # other floor
                self.assertTrue(active[3])                   # other building
                self.assertEqual(int(active[0]) + int(active[2]), 1)   # same floor, 5 m

    def test_other_topologies_keep_distance_only(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                # all four APs within 20 m of each other in 2D -> one winner
                self.assertEqual(_run("HOTSPOT", seed).sum(), 1)


if __name__ == "__main__":
    unittest.main()
