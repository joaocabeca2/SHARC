# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:56:10 2017

@author: Calil
"""

import unittest
import numpy as np

from sharc.mask.spectral_mask_imt import SpectralMaskImt
from sharc.support.enumerations import StationType


class SpectalMaskImtTest(unittest.TestCase):
    """Unit tests for the SpectralMaskImt class and its power calculation method."""

    def setUp(self):
        """Set up test cases for different station types and masks at various frequencies."""
        # Initialize variables for 40 GHz
        sta_type = StationType.IMT_BS
        p_tx = 25.1
        freq = 43000
        band = 200
        spurious = -13

        # Create mask for 40 GHz
        self.mask_bs_40GHz = SpectralMaskImt(sta_type, freq, band, spurious)
        self.mask_bs_40GHz.set_mask(p_tx)

        # Initialize variables for 40 GHz
        sta_type = StationType.IMT_BS
        p_tx = 28.1
        freq = 24350
        band = 200

        # Create mask for BS at 26 GHz
        self.mask_bs_26GHz = SpectralMaskImt(sta_type, freq, band, spurious)
        self.mask_bs_26GHz.set_mask(p_tx)

        # Create mask for UE at 26 GHz
        sta_type = StationType.IMT_UE
        self.mask_ue_26GHz = SpectralMaskImt(sta_type, freq, band, spurious)
        self.mask_ue_26GHz.set_mask(p_tx)

        # Initialize variables for 9GHz -13dBm/MHz
        sta_type = StationType.IMT_BS
        p_tx = 28.1
        freq = 9000
        band = 200

        # Create mask for BS at 9 GHz
        self.mask_bs_9GHz = SpectralMaskImt(sta_type, freq, band, -13)
        self.mask_bs_9GHz.set_mask(p_tx)

        # Initialize variables for 9GHz -30dBm/MHz
        sta_type = StationType.IMT_BS
        p_tx = 28.1
        freq = 9000
        band = 200

        # Create mask for BS at 9 GHz and spurious emission at -30dBm/MHz
        self.mask_bs_9GHz_30_spurious = SpectralMaskImt(
            sta_type, freq, band, -30,
        )
        self.mask_bs_9GHz_30_spurious.set_mask(p_tx)

    def test_power_calc(self):
        """Test power calculation for different masks and frequency/bandwidth combinations."""
        #######################################################################
        # Testing mask for 40 GHz
        #######################################################################

        # Test 1
        fc = 43000
        band = 200
        with self.assertWarns(RuntimeWarning):
            poob = self.mask_bs_40GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, -np.inf, delta=1e-2)

        # Test 2
        fc = 43300
        band = 600
        poob = self.mask_bs_40GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 11.8003, delta=1e-2)

        # Test 3
        fc = 43000
        band = 1200
        poob = self.mask_bs_40GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 14.8106, delta=1e-2)

        # Test 4
        fc = 45000
        band = 1000
        poob = self.mask_bs_40GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 17, delta=1e-2)

        #######################################################################
        # Testing mask for 26 GHz
        #######################################################################

        # Test 1 - BS
        fc = 23800
        band = 400
        poob = self.mask_bs_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 11.53, delta=1e-2)

        # Test 2 - BS
        fc = 23700
        band = 400
        poob = self.mask_bs_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 12.58, delta=1e-2)

        # Test 3 - BS
        fc = 23600
        band = 400
        poob = self.mask_bs_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 13.02, delta=1e-2)

        # Test 1 - UE
        fc = 23800
        band = 400
        poob = self.mask_ue_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 13.02, delta=1e-2)

        # Test 2 - UE
        fc = 23700
        band = 400
        poob = self.mask_ue_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 13.02, delta=1e-2)

        # Test 3 - UE
        fc = 23600
        band = 400
        poob = self.mask_ue_26GHz.power_calc(fc, band)
        self.assertAlmostEqual(poob, 13.02, delta=1e-2)


if __name__ == '__main__':
    unittest.main()
