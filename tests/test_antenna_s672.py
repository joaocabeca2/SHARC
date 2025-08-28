# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:58:50 2017

@author: edgar
"""

import unittest

from sharc.antenna.antenna_s672 import AntennaS672
from sharc.parameters.parameters_fss_ss import ParametersFssSs

import numpy as np
import numpy.testing as npt


class AntennaS672Test(unittest.TestCase):
    """Unit tests for the AntennaS672 class."""

    def setUp(self):
        """Set up test fixtures for AntennaS672 tests."""
        param = ParametersFssSs()
        param.antenna_gain = 50
        param.antenna_pattern = "ITU-R S.672-4"
        param.antenna_3_dB = 2

        param.antenna_l_s = -20
        self.antenna20 = AntennaS672(param)

        param.antenna_l_s = -30
        self.antenna30 = AntennaS672(param)

    def test_calculate_gain(self):
        """Test calculate_gain method for different antenna_l_s values."""
        psi = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])

        ref_gain20 = np.array(
            [0, -3, -12, -20, -20, -20, -20, -21.12, -22.57, -23.85, -25, -50],
        )
        gain20 = self.antenna20.calculate_gain(
            off_axis_angle_vec=psi,
        ) - self.antenna20.peak_gain
        npt.assert_allclose(gain20, ref_gain20, atol=1e-2)

        ref_gain30 = np.array(
            [0, -3, -12, -27, -30, -30, -30, -31.12, -32.57, -33.85, -35, -50],
        )
        gain30 = self.antenna30.calculate_gain(
            off_axis_angle_vec=psi,
        ) - self.antenna30.peak_gain
        npt.assert_allclose(gain30, ref_gain30, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
