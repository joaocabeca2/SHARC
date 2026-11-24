"""Regression checks for preserving the current Wi-Fi outputs during integration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from sharc.antenna.antenna_beamforming_wifi import AntennaBeamformingWifi
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.propagation.propagation_indoor_building import FloorPenetrationLoss
from sharc.results import Results
from sharc.simulation import Simulation
from sharc.simulation_downlink import SimulationDownlink
from sharc.simulation_uplink import SimulationUplink
from sharc.station_manager import StationManager
from sharc.system.system_wifi import SystemWifi


class WifiUpdateCompatibilityTest(unittest.TestCase):
    def test_csv_names_and_samples_column_are_preserved(self):
        samples = {
            "system_dl_interf_power": [-80.0],
            "system_ul_interf_power": [-81.0],
            "system_intra_dl_interf_power": [-90.0],
            "system_intra_ul_interf_power": [-91.0],
            "system_interf_power_total": [-79.5860731484],
            "imt_dl_interf_power_total": [-70.0],
            "imt_ul_interf_power_total": [-71.0],
            "system_dl_sinr": [12.0],
            "system_ul_sinr": [13.0],
            "system_inr": [-5.0],
        }
        result = Results()
        self.assertFalse(hasattr(result, "wifi_dl_sinr"))
        self.assertFalse(hasattr(result, "wifi_ul_sinr"))
        with tempfile.TemporaryDirectory() as directory:
            result.output_directory = directory
            result.overwrite_sample_files = True
            for name, values in samples.items():
                getattr(result, name).extend(values)
            result.write_files(1)
            self.assertEqual(
                {p.name for p in Path(directory).glob("*.csv")},
                {name + ".csv" for name in samples},
            )
            for name, values in samples.items():
                frame = pd.read_csv(Path(directory) / (name + ".csv"))
                self.assertEqual(list(frame.columns), ["samples"])
                np.testing.assert_allclose(frame["samples"].to_numpy(), values)

    def test_eirp_limit_is_optional_in_both_power_controls(self):
        for cls in (SimulationDownlink, SimulationUplink):
            for limited in (False, True):
                with self.subTest(simulation=cls.__name__, limited=limited):
                    param = ParametersWifiSystem()
                    param.ap.conducted_power = 23.0
                    param.sta.conducted_power = 23.0
                    param.ap.ohmic_loss = param.sta.ohmic_loss = 3.0
                    param.ap.max_eirp = 24.0 if limited else None
                    param.sta.max_eirp = 18.0 if limited else None
                    ap = SimpleNamespace(active=np.array([False, True]), spectral_mask=Mock())
                    sta = SimpleNamespace(active=np.array([True, False]), spectral_mask=Mock())
                    sim = SimpleNamespace(
                        parameters=SimpleNamespace(wifi=param),
                        ap_power_gain=0.0, sta_power_gain=0.0,
                        system=SimpleNamespace(ap=ap, sta=sta, ap_peak_gain=6.02),
                        wifi_conducted_power=Simulation.wifi_conducted_power,
                    )
                    cls.power_control_wifi(sim)
                    self.assertEqual(set(ap.tx_power), {1})
                    self.assertEqual(set(sta.tx_power), {0})
                    self.assertAlmostEqual(ap.tx_power[1], 20.98 if limited else 23.0)
                    self.assertAlmostEqual(sta.tx_power[0], 21.0 if limited else 23.0)
                    ap.spectral_mask.set_mask.assert_called_once_with(
                        p_tx=ap.tx_power[1])
                    sta.spectral_mask.set_mask.assert_called_once_with(
                        p_tx=sta.tx_power[0])

    def test_multi_sta_beamforming_is_rejected_until_direction_model_is_resolved(self):
        param = ParametersWifiSystem()
        param.sta.k = 2
        antenna = ParametersAntennaWifi()
        antenna.beamforming = True
        topology = SimpleNamespace(
            calculate_coordinates=lambda rng: None, num_base_stations=1)
        with self.assertRaisesRegex(ValueError, "one STA per AP"):
            SystemWifi(param, antenna, np.random.RandomState(7), topology)

    def test_single_sta_beam_is_pointed_at_the_associated_station(self):
        param = ParametersAntennaWifi(
            n_rows=1, n_columns=4, beamforming=True, downtilt=0.0)
        ap, sta = StationManager(1), StationManager(1)
        ap.x, ap.y, ap.height = np.array([0.0]), np.array([0.0]), np.array([3.0])
        sta.x, sta.y, sta.height = np.array([10.0]), np.array([10.0]), np.array([1.5])
        ap.active = np.array([True])
        ap.antenna[0] = AntennaBeamformingWifi(param, 0.0, 0.0)
        system = SimpleNamespace(ap=ap, sta=sta, link={0: [0]})
        SystemWifi.point_ap_beams(system)
        phi, theta = ap.get_pointing_vector_to(sta)
        beam_phi, beam_theta = ap.antenna[0].beams_list[0]
        expected_phi, expected_theta = ap.antenna[0].to_local_coord(phi[0, 0], theta[0, 0])
        self.assertAlmostEqual(beam_phi, float(expected_phi[0]))
        self.assertAlmostEqual(beam_theta, float(expected_theta[0] - 90.0))

    def test_floor_loss_matches_scalar_rule_and_handles_empty_inputs(self):
        floors = np.array([[0, 1, 2], [3, 4, 8]])
        for frequency in (np.full(floors.shape, 7000.0),
                          np.array([[900.0, 1800.0, 5200.0], [7000.0, 8000.0, 26000.0]])):
            for environment in ("OFFICE", "RESIDENTIAL", "COMMERCIAL"):
                with self.subTest(environment=environment):
                    expected = np.array([
                        FloorPenetrationLoss._loss_scalar(f, int(n), environment)
                        for f, n in zip(frequency.ravel(), floors.ravel())
                    ]).reshape(floors.shape)
                    np.testing.assert_allclose(
                        FloorPenetrationLoss.get_loss(frequency, floors, environment),
                        expected,
                    )
        self.assertEqual(
            FloorPenetrationLoss.get_loss(np.empty((0, 2)), np.empty((0, 2))).shape,
            (0, 2),
        )


if __name__ == "__main__":
    unittest.main()

