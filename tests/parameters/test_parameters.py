import unittest
from sharc.parameters.parameters import Parameters


class ParametersTest(unittest.TestCase):

    def setUp(self):
        self.parameters = Parameters()
        self.parameters.set_file_name("./tests/parameters/parameters_for_testing.ini")
        self.parameters.read_params()

    def test_parameters_imt(self):

        self.assertEqual(self.parameters.imt.topology, "INDOOR")
        self.assertEqual(self.parameters.imt.wrap_around, False)
        self.assertEqual(self.parameters.imt.num_clusters, 1)
        self.assertEqual(self.parameters.imt.intersite_distance, 399)
        self.assertEqual(self.parameters.imt.minimum_separation_distance_bs_ue, 1.3)
        self.assertEqual(self.parameters.imt.interfered_with, False)
        self.assertEqual(self.parameters.imt.frequency, 24360)
        self.assertEqual(self.parameters.imt.bandwidth, 200.5)
        self.assertEqual(self.parameters.imt.rb_bandwidth, 0.181)
        self.assertEqual(self.parameters.imt.spectral_mask, "3GPP E-UTRA")
        self.assertEqual(self.parameters.imt.spurious_emissions, -13.1)
        self.assertEqual(self.parameters.imt.guard_band_ratio, 0.14)
        self.assertEqual(self.parameters.imt.bs_load_probability, 0.2)
        self.assertEqual(self.parameters.imt.bs_conducted_power, 11.1)
        self.assertEqual(self.parameters.imt.bs_height, 6.1)
        self.assertEqual(self.parameters.imt.bs_noise_figure, 10.1)
        self.assertEqual(self.parameters.imt.bs_noise_temperature, 290.1)
        self.assertEqual(self.parameters.imt.bs_ohmic_loss, 3.1)
        self.assertEqual(self.parameters.imt.ul_attenuation_factor, 0.4)
        self.assertEqual(self.parameters.imt.ul_sinr_min, -10.0)
        self.assertEqual(self.parameters.imt.ul_sinr_max, 22.0)
        self.assertEqual(self.parameters.imt.ue_k, 3)
        self.assertEqual(self.parameters.imt.ue_k_m, 1)
        self.assertEqual(self.parameters.imt.ue_indoor_percent, 5.0)
        self.assertEqual(self.parameters.imt.ue_distribution_type, "ANGLE_AND_DISTANCE")
        self.assertEqual(self.parameters.imt.ue_distribution_distance, "UNIFORM")
        self.assertEqual(self.parameters.imt.ue_tx_power_control, True)
        self.assertEqual(self.parameters.imt.ue_p_o_pusch, -95.0)
        self.assertEqual(self.parameters.imt.ue_alpha, 1.0)
        self.assertEqual(self.parameters.imt.ue_p_cmax, 22.0)
        self.assertEqual(self.parameters.imt.ue_power_dynamic_range, 63.0)
        self.assertEqual(self.parameters.imt.ue_height, 1.5)
        self.assertEqual(self.parameters.imt.ue_noise_figure, 10.0)
        self.assertEqual(self.parameters.imt.ue_ohmic_loss, 3.0)
        self.assertEqual(self.parameters.imt.ue_body_loss, 4.0)
        self.assertEqual(self.parameters.imt.dl_attenuation_factor, 0.6)
        self.assertEqual(self.parameters.imt.dl_sinr_min, -10.0)
        self.assertEqual(self.parameters.imt.dl_sinr_max, 30.0)
        self.assertEqual(self.parameters.imt.channel_model, "FSPL")
        self.assertEqual(self.parameters.imt.los_adjustment_factor, 18.0)
        self.assertEqual(self.parameters.imt.shadowing, False)

    # def test_parameters_imt_antenna(self):
    #     pass

    # def test_parameters_hotpot(self):
    #     pass

    # def test_parameters_indoor(self):
    #     pass

    # def test_parameters_fss_ss(self):
    #     pass

    # def test_parameters_fss_es(self):
    #     pass

    # def test_parameters_fs(self):
    #     pass

    # def test_parameters_haps(self):
    #     pass

    # def test_paramters_rns(self):
    #     pass

    # def test_parametes_ras(self):
    #     pass
    
    # def test_paramters_eess_passive(self):
    #     pass

if __name__ == '__main__':
    unittest.main()