import yaml
import unittest
from sharc.parameters.parameters import Parameters
class Parameters:
    def __init__(self):
        self.file_name = None
        self.imt = None
        self.antenna_imt = None
        self.hotspot = None
        self.indoor = None
        self.fss_ss = None
        self.fss_es = None
        self.fs = None
        self.haps = None
        self.rns = None
        self.ras = None
        self.eess_passive = None

    def set_file_name(self, file_name):
        self.file_name = 'tests/parameters/parameters_for_testing.yaml'

    def read_params(self):
        with open(self.file_name, 'r') as file:
            params = yaml.safe_load(file)
            self.imt = params.get('imt')
            self.antenna_imt = params.get('antenna_imt')
            self.hotspot = params.get('hotspot')
            self.indoor = params.get('indoor')
            self.fss_ss = params.get('fss_ss')
            self.fss_es = params.get('fss_es')
            self.fs = params.get('fs')
            self.haps = params.get('haps')
            self.rns = params.get('rns')
            self.ras = params.get('ras')
            self.eess_passive = params.get('eess_passive')   

class ParametersTest(unittest.TestCase):
    """Run Parameter class tests.
    """
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.set_file_name("./parameters/parameters_for_testing.yaml")
        self.parameters.set_file_name("./parameters/parameters_for_testing.yaml")
        self.parameters.read_params()

    def test_parameters_imt(self):
        """Unit test for ParametersIMT
        """
        self.assertEqual(self.parameters.imt['topology'], "INDOOR")
        self.assertEqual(self.parameters.imt['wrap_around'], False)
        self.assertEqual(self.parameters.imt['num_clusters'], 1)
        self.assertEqual(self.parameters.imt['intersite_distance'], 399)
        self.assertEqual(self.parameters.imt['minimum_separation_distance_bs_ue'], 1.3)
        self.assertEqual(self.parameters.imt['interfered_with'], False)
        self.assertEqual(self.parameters.imt['frequency'], 24360)
        self.assertEqual(self.parameters.imt['bandwidth'], 200.5)
        self.assertEqual(self.parameters.imt['rb_bandwidth'], 0.181)
        self.assertEqual(self.parameters.imt['spectral_mask'], "3GPP E-UTRA")
        self.assertEqual(self.parameters.imt['spurious_emissions'], -13.1)
        self.assertEqual(self.parameters.imt['guard_band_ratio'], 0.14)
        self.assertEqual(self.parameters.imt['bs_load_probability'], 0.2)
        self.assertEqual(self.parameters.imt['bs_conducted_power'], 11.1)
        self.assertEqual(self.parameters.imt['bs_height'], 6.1)
        self.assertEqual(self.parameters.imt['bs_noise_figure'], 10.1)
        self.assertEqual(self.parameters.imt['bs_noise_temperature'], 290.1)
        self.assertEqual(self.parameters.imt['bs_ohmic_loss'], 3.1)
        self.assertEqual(self.parameters.imt['ul_attenuation_factor'], 0.4)
        self.assertEqual(self.parameters.imt['ul_sinr_min'], -10.0)
        self.assertEqual(self.parameters.imt['ul_sinr_max'], 22.0)
        self.assertEqual(self.parameters.imt['ue_k'], 3)
        self.assertEqual(self.parameters.imt['ue_k_m'], 1)
        self.assertEqual(self.parameters.imt['ue_indoor_percent'], 5.0)
        self.assertEqual(self.parameters.imt['ue_distribution_type'], "ANGLE_AND_DISTANCE")
        self.assertEqual(self.parameters.imt['ue_distribution_distance'], "UNIFORM")
        self.assertEqual(self.parameters.imt['ue_tx_power_control'], True)
        self.assertEqual(self.parameters.imt['ue_p_o_pusch'], -95.0)
        self.assertEqual(self.parameters.imt['ue_alpha'], 1.0)
        self.assertEqual(self.parameters.imt['ue_p_cmax'], 22.0)
        self.assertEqual(self.parameters.imt['ue_power_dynamic_range'], 63.0)
        self.assertEqual(self.parameters.imt['ue_height'], 1.5)
        self.assertEqual(self.parameters.imt['ue_noise_figure'], 10.0)
        self.assertEqual(self.parameters.imt['ue_ohmic_loss'], 3.0)
        self.assertEqual(self.parameters.imt['ue_body_loss'], 4.0)
        self.assertEqual(self.parameters.imt['dl_attenuation_factor'], 0.6)

    def test_parameters_single_earth_station(self):
        """Test ParametersSingleEarthStation
        """
        self.assertEqual(self.parameters.single_earth_station.frequency, 8250)
        self.assertEqual(self.parameters.single_earth_station.bandwidth, 100)
        self.assertEqual(self.parameters.single_earth_station.adjacent_ch_selectivity, 20.0)
        self.assertEqual(self.parameters.single_earth_station.tx_power_density, -65.0)
        self.assertEqual(self.parameters.single_earth_station.noise_temperature, 300)
        self.assertEqual(self.parameters.single_earth_station.geometry.height, 6)
        self.assertEqual(self.parameters.single_earth_station.geometry.azimuth.type, "FIXED")
        self.assertEqual(self.parameters.single_earth_station.geometry.azimuth.fixed, 0)
        self.assertEqual(self.parameters.single_earth_station.geometry.azimuth.uniform_dist.min, -180)
        self.assertEqual(self.parameters.single_earth_station.geometry.azimuth.uniform_dist.max, 180)
        self.assertEqual(self.parameters.single_earth_station.geometry.elevation.type, "FIXED")
        self.assertEqual(self.parameters.single_earth_station.geometry.elevation.fixed, 60)
        self.assertEqual(self.parameters.single_earth_station.geometry.elevation.uniform_dist.min, 30)
        self.assertEqual(self.parameters.single_earth_station.geometry.elevation.uniform_dist.max, 65)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.type, "CELL")
        self.assertEqual(self.parameters.single_earth_station.geometry.location.fixed.x, 10)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.fixed.y, 100)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.uniform_dist.min_dist_to_center, 101)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.uniform_dist.max_dist_to_center, 102)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.cell.min_dist_to_bs, 100)
        self.assertEqual(self.parameters.single_earth_station.geometry.location.network.min_dist_to_bs, 150)
        self.assertEqual(self.parameters.single_earth_station.antenna.gain, 28)
        self.assertEqual(self.parameters.single_earth_station.antenna.itu_r_f_699.diameter, 1.1)
        self.assertEqual(self.parameters.single_earth_station.antenna.itu_r_f_699.frequency, self.parameters.single_earth_station.frequency)
        self.assertEqual(self.parameters.single_earth_station.antenna.itu_r_f_699.antenna_gain, self.parameters.single_earth_station.antenna.gain)

        self.assertEqual(self.parameters.single_earth_station.param_p619.earth_station_alt_m, 1200)
        self.assertEqual(self.parameters.single_earth_station.param_p619.space_station_alt_m, 540000)
        self.assertEqual(self.parameters.single_earth_station.param_p619.earth_station_lat_deg, 13)
        self.assertEqual(self.parameters.single_earth_station.param_p619.earth_station_long_diff_deg, 10)

        
        self.assertEqual(self.parameters.single_earth_station.param_p452.atmospheric_pressure, 1)
        self.assertEqual(self.parameters.single_earth_station.param_p452.air_temperature, 2)
        self.assertEqual(self.parameters.single_earth_station.param_p452.N0, 3)
        self.assertEqual(self.parameters.single_earth_station.param_p452.delta_N, 4)
        self.assertEqual(self.parameters.single_earth_station.param_p452.percentage_p, 5)
        self.assertEqual(self.parameters.single_earth_station.param_p452.Dct, 6)
        self.assertEqual(self.parameters.single_earth_station.param_p452.Dcr, 7)
        self.assertEqual(self.parameters.single_earth_station.param_p452.Hte, 8)
        self.assertEqual(self.parameters.single_earth_station.param_p452.Hre, 9)
        self.assertEqual(self.parameters.single_earth_station.param_p452.tx_lat, 10)
        self.assertEqual(self.parameters.single_earth_station.param_p452.rx_lat, 11)
        self.assertEqual(self.parameters.single_earth_station.param_p452.polarization, "horizontal")
        self.assertEqual(self.parameters.single_earth_station.param_p452.clutter_loss, True)

        self.assertEqual(self.parameters.single_earth_station.param_hdfss.es_position, "BUILDINGSIDE")
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.shadow_enabled, False)
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.building_loss_enabled, False)
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.same_building_enabled, True)
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.diffraction_enabled, False)
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.bs_building_entry_loss_type, "FIXED_VALUE")
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.bs_building_entry_loss_prob, 0.19)
        self.assertEqual(self.parameters.single_earth_station.param_hdfss.bs_building_entry_loss_value, 47)

        self.parameters.single_earth_station.geometry.azimuth.uniform_dist.max = None
        # this should still not throw, since azimuth is using fixed type
        self.parameters.single_earth_station.validate()

        # now it should throw:
        with self.assertRaises(ValueError) as err_context:
            self.parameters.single_earth_station.geometry.azimuth.type = "UNIFORM_DIST"
            self.parameters.single_earth_station.validate()

        self.assertTrue('azimuth.uniform_dist.max' in str(err_context.exception))

if __name__ == '__main__':
    unittest.main()