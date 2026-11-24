"""Half-duplex/CCA regression tests for the existing CSV names and node indices."""
import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import yaml

from sharc.antenna.antenna_beamforming_wifi import AntennaBeamformingWifi
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.parameters.parameters import Parameters
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.results import Results
from sharc.simulation_downlink import SimulationDownlink
from sharc.simulation_uplink import SimulationUplink
from sharc.station_manager import StationManager
from sharc.system.system_wifi import SystemWifi


def simulation(cls, n_ap=4, n_sta=8):
    sim = cls.__new__(cls)
    sim.parameters = Parameters()
    sim.parameters.general.system = "WIFI"
    sim.parameters.general.imt_link = "DOWNLINK" if cls is SimulationDownlink else "UPLINK"
    sim.parameters.wifi.prob_DL = 0.5
    sim.param_system = sim.parameters.wifi
    sim.ap_power_gain = sim.sta_power_gain = 0.0
    system = SystemWifi.__new__(SystemWifi)
    system.parameters = sim.param_system
    system.num_aps, system.num_sta = n_ap, n_sta
    system.ap, system.sta = StationManager(n_ap), StationManager(n_sta)
    system.link = {a: list(range(a * (n_sta // n_ap), (a + 1) * (n_sta // n_ap))) for a in range(n_ap)}
    for manager in (system.ap, system.sta):
        manager.antenna = np.array([AntennaOmni(0) for _ in range(manager.num_stations)], dtype=object)
        manager.noise_figure = np.arange(manager.num_stations, dtype=float) + 3.0
        manager.spectral_mask = Mock()
    sim.system = system
    sim.bs = SimpleNamespace(active=np.array([], dtype=bool))
    sim.results = Results()
    return sim


def channel(sim):
    ap, sta = sim.system.ap, sim.system.sta
    sim.coupling_loss_wifi = np.full((ap.num_stations, sta.num_stations), 100.0)
    sim.coupling_loss_ap_ap = np.full((ap.num_stations, ap.num_stations), 90.0)
    sim.coupling_loss_sta_sta = np.full((sta.num_stations, sta.num_stations), 85.0)
    np.fill_diagonal(sim.coupling_loss_ap_ap, np.inf)
    np.fill_diagonal(sim.coupling_loss_sta_sta, np.inf)
    sim.path_loss_wifi = sim.coupling_loss_wifi.copy()
    sim.path_loss_ap_ap = sim.coupling_loss_ap_ap.copy()
    sim.path_loss_sta_sta = sim.coupling_loss_sta_sta.copy()
    sim.ap_antenna_gain = np.ones_like(sim.coupling_loss_wifi)
    sim.sta_antenna_gain = np.full_like(sim.coupling_loss_wifi, 2.0)


class WifiLinkResultsTest(unittest.TestCase):
    def test_all_collector_sample_fields_exist_with_current_names(self):
        result = Results()
        for cls in (SimulationDownlink, SimulationUplink):
            source = ast.parse(inspect.getsource(inspect.getmodule(cls)))
            methods = [node for node in ast.walk(source)
                       if isinstance(node, ast.FunctionDef)
                       and node.name in ("collect_results", "collect_results_wifi")]
            for method in methods:
                for field in ast.walk(method):
                    if (isinstance(field, ast.Attribute)
                            and isinstance(field.value, ast.Attribute)
                            and isinstance(field.value.value, ast.Name)
                            and field.value.value.id == "self"
                            and field.value.attr == "results"):
                        self.assertTrue(hasattr(result, field.attr), field.attr)

    def test_original_node_ids_and_csv_meanings_in_dl_and_ul(self):
        for cls, short in ((SimulationDownlink, "dl"), (SimulationUplink, "ul")):
            with self.subTest(direction=short):
                sim = simulation(cls)
                sim.system.set_links_state([(1, 3, True), (3, 7, False)])
                sim.system.ap.tx_power = {1: 20.0}
                sim.system.sta.tx_power = {7: 10.0}
                channel(sim)
                sim.coupling_loss_wifi[1, 3] = 60.0
                sim.coupling_loss_wifi[3, 7] = 70.0
                sim.calculate_sinr_wifi(reuse_coupling_loss=True)
                self.assertEqual(sim.system.ap.sinr.shape, (4,))
                self.assertEqual(sim.system.sta.sinr.shape, (8,))
                self.assertAlmostEqual(sim.system.ap.rx_power[3], -60.0)
                self.assertAlmostEqual(sim.system.sta.rx_power[3], -40.0)
                self.assertAlmostEqual(sim.system.ap.rx_interference[3], -70.0)
                self.assertAlmostEqual(sim.system.sta.rx_interference[3], -75.0)
                ap_lin, sta_lin = np.zeros(4), np.zeros(8)
                ap_lin[3], sta_lin[3] = 10 ** -8, 10 ** -6.5
                sim.imt_to_wifi_linear = lambda a, s: (ap_lin, sta_lin)
                sim.calculate_external_interference_wifi()
                sim.collect_results_wifi(False, 1)
                np.testing.assert_allclose(
                    getattr(sim.results, "system_" + short + "_interf_power"), [-80.0, -65.0])
                np.testing.assert_allclose(
                    getattr(sim.results, "system_intra_" + short + "_interf_power"), [-70.0, -75.0])
                total = np.array(sim.results.system_interf_power_total)
                np.testing.assert_allclose(
                    10 ** (total / 10), np.array([10 ** -7 + 10 ** -8, 10 ** -7.5 + 10 ** -6.5]))
                for manager in (sim.system.ap, sim.system.sta):
                    index = 3
                    expected = manager.rx_power[index] - 10 * np.log10(
                        10 ** (manager.intra_interference[index] / 10)
                        + 10 ** (manager.ext_interference[index] / 10)
                        + 10 ** (manager.thermal_noise[index] / 10))
                    self.assertAlmostEqual(manager.sinr_ext[index], expected)
                self.assertEqual(len(getattr(sim.results, "system_" + short + "_sinr")), 2)
                self.assertEqual(len(getattr(sim.results, "system_" + short + "_sinr_ext")), 2)
                self.assertEqual(len(sim.results.system_path_loss) > 0, True)
                np.testing.assert_allclose(sim.results.system_ap_antenna_gain, [1.0, 1.0])
                np.testing.assert_allclose(sim.results.system_sta_antenna_gain, [2.0, 2.0])
                with tempfile.TemporaryDirectory() as directory:
                    sim.results.output_directory = directory
                    sim.results.overwrite_sample_files = True
                    sim.results.write_files(1)
                    paths = {p.name for p in Path(directory).glob("*.csv")}
                    self.assertIn("system_interf_power_total.csv", paths)
                    self.assertIn("system_" + short + "_interf_power.csv", paths)
                    self.assertNotIn("system_ext_" + short + "_interf_power.csv", paths)
                    self.assertFalse(any(p.startswith("wifi_") for p in paths))
                    frame = pd.read_csv(Path(directory) / ("system_" + short + "_interf_power.csv"))
                    self.assertEqual(list(frame.columns), ["samples"])

    def test_cca_with_sta_id_greater_than_ap_count(self):
        sim = simulation(SimulationDownlink, 2, 6)
        sim.parameters.wifi.cca_mode = "POWER"
        sim.parameters.wifi.cca_detect_imt = False
        sim.system.draw_csma_links = Mock(return_value=[(0, 2, True), (1, 5, False)])
        channel(sim)
        def coupling(a, b):
            if a is b:
                out = np.full((a.num_stations, a.num_stations), 100.0)
                np.fill_diagonal(out, np.inf)
                return out
            return np.full((b.num_stations, a.num_stations), 100.0)
        sim.calculate_intra_wifi_coupling_loss = coupling
        sim.run_wifi_cca_power(np.random.RandomState(7))
        np.testing.assert_array_equal(sim.system.sta_tx, [5])
        np.testing.assert_array_equal(sim.system.sta_rx, [2])
        self.assertEqual(sim.system.sta.sinr.shape, (6,))
        self.assertEqual(sim.results.system_cca_links_intent, [2])
        self.assertEqual(sim.results.system_cca_links_tx, [2])

    def test_all_cca_links_blocked_has_no_receiver_samples(self):
        sim = simulation(SimulationDownlink, 2, 6)
        sim.parameters.wifi.cca_mode = "POWER"
        sim.system.draw_csma_links = Mock(return_value=[(0, 2, True), (1, 5, False)])
        channel(sim)
        sim.calculate_intra_wifi_coupling_loss = lambda a, b: np.full(
            (b.num_stations, a.num_stations), 100.0)
        sim.imt_power_at_wifi_nodes = lambda: (np.full(2, -20.0), np.full(6, -20.0))
        sim.run_wifi_cca_power(np.random.RandomState(7))
        self.assertEqual(len(sim.system.ap_rx) + len(sim.system.sta_rx), 0)
        self.assertEqual(sim.results.system_cca_links_tx, [0])
        self.assertEqual(sim.results.system_cca_links_blocked_imt, [2])
        self.assertEqual(sim.system.ap.sinr.shape, (2,))
        self.assertEqual(sim.system.sta.sinr.shape, (6,))

    def test_half_duplex_does_not_divide_power_by_unscheduled_stas(self):
        for cls in (SimulationDownlink, SimulationUplink):
            sim = simulation(cls)
            sim.parameters.wifi.sta.k = 4
            sim.system.set_links_state([(1, 3, True)])
            sim.power_control_wifi()
            self.assertEqual(sim.system.ap.tx_power[1], sim.parameters.wifi.ap.conducted_power)
            self.assertEqual(sim.system.sta.tx_power, {})

    def test_cca_boolean_string_false_is_not_truthy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.yaml"
            path.write_text(yaml.safe_dump({"wifi": {
                "prob_DL": 0.5, "cca_mode": "POWER", "cca_detect_imt": "false"}}))
            param = ParametersWifiSystem()
            param.load_parameters_from_file(str(path))
            self.assertFalse(param.cca_detect_imt)

    def test_channel_cache_reuses_realization_and_restores_metadata(self):
        sim = simulation(SimulationDownlink)
        sim.parameters.wifi.cca_mode = "POWER"
        other = StationManager(2)
        def calculate(*args):
            sim.imt_system_path_loss = np.array([[123.0]])
            return np.array([[100.0]])
        sim._calculate_coupling_loss_system_imt_uncached = Mock(side_effect=calculate)
        first = sim.calculate_coupling_loss_system_imt(sim.system.ap, other)
        sim.imt_system_path_loss = np.array([[999.0]])
        second = sim.calculate_coupling_loss_system_imt(sim.system.ap, other)
        self.assertIs(first, second)
        self.assertEqual(sim.imt_system_path_loss[0, 0], 123.0)
        self.assertEqual(sim._calculate_coupling_loss_system_imt_uncached.call_count, 1)
        sim.clear_wifi_coupling_cache(sim.system.ap)
        sim.calculate_coupling_loss_system_imt(sim.system.ap, other)
        self.assertEqual(sim._calculate_coupling_loss_system_imt_uncached.call_count, 2)


    def test_beam_points_to_the_selected_nonfirst_sta(self):
        sim = simulation(SimulationDownlink, 1, 2)
        param = ParametersAntennaWifi(n_rows=1, n_columns=4, beamforming=True, downtilt=0.0)
        ap, sta = sim.system.ap, sim.system.sta
        ap.x, ap.y, ap.height = np.array([0.0]), np.array([0.0]), np.array([3.0])
        sta.x, sta.y, sta.height = np.array([10.0, 0.0]), np.array([0.0, 10.0]), np.array([1.5, 1.5])
        ap.antenna[0] = AntennaBeamformingWifi(param, 0.0, 0.0)
        sim.system.set_links_state([(0, 1, True)])
        sim.system.point_ap_beams()
        phi, theta = ap.get_pointing_vector_to(sta)
        expected_phi, expected_theta = ap.antenna[0].to_local_coord(phi[0, 1], theta[0, 1])
        self.assertAlmostEqual(ap.antenna[0].beams_list[0][0], expected_phi[0])
        self.assertAlmostEqual(ap.antenna[0].beams_list[0][1], expected_theta[0] - 90.0)

    def test_uplink_gain_csv_uses_beam_columns_and_tx_rx_node_ids(self):
        sim = simulation(SimulationUplink)
        sim.system.set_links_state([(1, 3, True), (3, 7, False)])
        sim.system.ap.tx_power, sim.system.sta.tx_power = {1: 20.0}, {7: 10.0}
        channel(sim)
        sim.calculate_sinr_wifi(reuse_coupling_loss=True)
        sim.imt_to_wifi_linear = lambda a, s: (np.zeros(4), np.zeros(8))
        sim.calculate_external_interference_wifi()
        sim.parameters.imt.ue.k = 2
        sim.link = {1: [6, 7]}
        sim.bs = StationManager(2)
        sim.bs.active = np.array([False, True])
        for field in ("sinr", "sinr_ext", "snr", "inr", "rx_interference",
                      "ext_interference", "interf_power_total"):
            setattr(sim.bs, field, {1: np.array([1.0, 2.0])})
        sim.ue = StationManager(8)
        sim.ue.tx_power = np.arange(8, dtype=float)
        sim.path_loss_imt = sim.coupling_loss_imt = np.ones((2, 8))
        sim.imt_bs_antenna_gain = sim.imt_ue_antenna_gain = np.ones((2, 8))
        sim.ap_imt_antenna_gain = np.arange(16.0).reshape(4, 4)
        sim.sta_imt_antenna_gain = np.arange(32.0).reshape(8, 4)
        sim.imt_ap_antenna_gain = 100.0 + np.arange(16.0).reshape(4, 4)
        sim.imt_sta_antenna_gain = 200.0 + np.arange(32.0).reshape(8, 4)
        sim.imt_ap_path_loss = np.ones((4, 4))
        sim.imt_sta_path_loss = np.ones((8, 4))
        sim.collect_results_wifi(False, 1)
        np.testing.assert_allclose(sim.results.ap_imt_antenna_gain, [6.0, 7.0])
        np.testing.assert_allclose(sim.results.sta_imt_antenna_gain, [30.0, 31.0])
        np.testing.assert_allclose(sim.results.imt_ap_antenna_gain, [114.0, 115.0])
        np.testing.assert_allclose(sim.results.imt_sta_antenna_gain, [214.0, 215.0])


if __name__ == "__main__":
    unittest.main()

