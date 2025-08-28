# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:38:33 2018

@author: Calil
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.parameters.parameters_hdfss import ParametersHDFSS
from sharc.support.enumerations import StationType
from sharc.propagation.propagation_hdfss_roof_top import PropagationHDFSSRoofTop


class PropagationHDFSSRoofTopTest(unittest.TestCase):
    """Unit tests for the PropagationHDFSSRoofTop class and its loss calculations."""

    def setUp(self):
        """Set up test fixtures for PropagationHDFSSRoofTop tests."""
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = False
        par.shadow_enabled = False
        par.same_building_enabled = True
        par.diffraction_enabled = False
        par.bs_building_entry_loss_type = 'FIXED_VALUE'
        par.bs_building_entry_loss_prob = 0.5
        par.bs_building_entry_loss_value = 50
        self.propagation = PropagationHDFSSRoofTop(par, rnd)

        # Propagation with fixed BEL
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = True
        par.shadow_enabled = False
        par.same_building_enabled = True
        par.diffraction_enabled = False
        par.bs_building_entry_loss_type = 'FIXED_VALUE'
        par.bs_building_entry_loss_prob = 0.6
        par.bs_building_entry_loss_value = 50
        self.propagation_fixed_value = PropagationHDFSSRoofTop(par, rnd)

        # Propagation with fixed probability
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = True
        par.shadow_enabled = False
        par.same_building_enabled = True
        par.diffraction_enabled = False
        par.bs_building_entry_loss_type = 'P2109_FIXED'
        par.bs_building_entry_loss_prob = 0.6
        par.bs_building_entry_loss_value = 50
        self.propagation_fixed_prob = PropagationHDFSSRoofTop(par, rnd)

        # Propagation with random probability
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = True
        par.shadow_enabled = False
        par.same_building_enabled = True
        par.diffraction_enabled = False
        par.bs_building_entry_loss_type = 'P2109_RANDOM'
        par.bs_building_entry_loss_prob = 0.6
        par.bs_building_entry_loss_value = 50
        self.propagation_random_prob = PropagationHDFSSRoofTop(par, rnd)

        # Same building disabled
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = False
        par.shadow_enabled = False
        par.same_building_enabled = False
        par.diffraction_enabled = False
        par.bs_building_entry_loss_type = 'FIXED_VALUE'
        par.bs_building_entry_loss_prob = 0.5
        par.bs_building_entry_loss_value = 50
        self.propagation_same_build_disabled = PropagationHDFSSRoofTop(
            par, rnd,
        )

        # Diffraction loss enabled
        rnd = np.random.RandomState(101)
        par = ParametersHDFSS()
        par.building_loss_enabled = False
        par.shadow_enabled = False
        par.same_building_enabled = True
        par.diffraction_enabled = True
        par.bs_building_entry_loss_type = 'FIXED_VALUE'
        par.bs_building_entry_loss_prob = 0.5
        par.bs_building_entry_loss_value = 50
        self.propagation_diff_enabled = PropagationHDFSSRoofTop(par, rnd)

    def test_get_loss(self):
        """Test the get_loss method for various roof top scenarios."""
        # Not on same building
        d = np.array([[10.0, 20.0, 30.0, 60.0, 90.0, 300.0, 1000.0]])
        f = 40000 * np.ones_like(d)
        ele = np.transpose(np.zeros_like(d))

        loss = self.propagation.get_loss(
            distance_3D=d,
            frequency=f,
            elevation=ele,
            imt_sta_type=StationType.IMT_BS,
            imt_x=100.0 * np.ones(7),
            imt_y=100.0 * np.ones(7),
            imt_z=100.0 * np.ones(7),
            es_x=np.array([0.0]),
            es_y=np.array([0.0]),
            es_z=np.array([0.0]),
        )
        loss = loss[0]

        expected_loss = np.array(
            [[84.48, 90.50, 94.02, 100.72, 104.75, 139.33, 162.28]],
        )

        npt.assert_allclose(loss, expected_loss, atol=1e-1)

        # On same building
        d = np.array([[10.0, 20.0, 30.0]])
        f = 40000 * np.ones_like(d)
        ele = np.transpose(np.zeros_like(d))
        es_x = np.array([0.0])
        es_y = np.array([0.0])
        es_z = np.array([10.0])
        imt_x = np.array([0.0, 20.0, 30.0])
        imt_y = np.array([10.0, 0.0, 0.0])
        imt_z = np.array([1.5, 6.0, 7.5])

        loss = self.propagation.get_loss(
            distance_3D=d,
            frequency=f,
            elevation=ele,
            imt_sta_type=StationType.IMT_BS,
            imt_x=imt_x,
            imt_y=imt_y,
            imt_z=imt_z,
            es_x=es_x,
            es_y=es_y,
            es_z=es_z,
        )
        loss = loss[0]

        expected_loss = np.array([[150 + 84.48, 100 + 90.50, 50 + 94.02]])

        npt.assert_allclose(loss, expected_loss, atol=1e-1)

    def test_get_build_loss(self):
        """Test get_building_loss for various station types and probabilities."""
        # Initialize variables
        ele = np.array([[0.0, 45.0, 90.0]])
        f = 40000 * np.ones_like(ele)
        sta_type = StationType.IMT_BS

        # Test 1: fixed value
        expected_build_loss = 50.0
        build_loss = self.propagation_fixed_value.get_building_loss(
            sta_type,
            f,
            ele,
        )
        self.assertEqual(build_loss, expected_build_loss)

        # Test 2: fixed probability
        expected_build_loss = np.array([[24.4, 33.9, 43.4]])
        build_loss = self.propagation_fixed_prob.get_building_loss(
            sta_type,
            f,
            ele,
        )
        npt.assert_allclose(build_loss, expected_build_loss, atol=1e-1)

        # Test 3: random probability
        expected_build_loss = np.array([[21.7, 32.9, 15.9]])
        build_loss = self.propagation_random_prob.get_building_loss(
            sta_type,
            f,
            ele,
        )
        npt.assert_allclose(build_loss, expected_build_loss, atol=1e-1)

        # Test 4: UE station
        sta_type = StationType.IMT_UE
        expected_build_loss = np.array([[21.7, 32.9, 15.9]])
        build_loss = self.propagation_fixed_value.get_building_loss(
            sta_type,
            f,
            ele,
        )
        npt.assert_allclose(build_loss, expected_build_loss, atol=1e-1)
        build_loss = self.propagation_fixed_prob.get_building_loss(
            sta_type,
            f,
            ele,
        )
        npt.assert_allclose(build_loss, expected_build_loss, atol=1e-1)
        expected_build_loss = np.array([[10.1, 36.8, 52.6]])
        build_loss = self.propagation_random_prob.get_building_loss(
            sta_type,
            f,
            ele,
        )
        npt.assert_allclose(build_loss, expected_build_loss, atol=1e-1)

    def test_same_building(self):
        """Test is_same_building method for correct building identification and loss."""
        # Test is_same_building()
        es_x = np.array([0.0])
        es_y = np.array([0.0])
        es_z = np.array([19.0])
        imt_x = np.array([1.0, 0.0, 80.0, -70.0, 12.0])
        imt_y = np.array([1.0, 30.0, 0.0, -29.3, -3.6])
        imt_z = 3 * np.ones_like(imt_x)

        expected_in_build = np.array([True, False, False, False, True])
        in_build = self.propagation_same_build_disabled.is_same_building(
            imt_x,
            imt_y,
            es_x,
            es_y,
        )
        npt.assert_array_equal(in_build, expected_in_build)

        # Test loss
        d = np.sqrt(np.power(imt_x, 2) + np.power(imt_y, 2))
        d = np.array([list(d)])
        f = 40000 * np.ones_like(d)
        ele = np.transpose(np.zeros_like(d))

        loss = self.propagation_same_build_disabled.get_loss(
            distance_3D=d,
            frequency=f,
            elevation=ele,
            imt_sta_type=StationType.IMT_BS,
            imt_x=imt_x,
            imt_y=imt_y,
            imt_z=imt_z,
            es_x=es_x,
            es_y=es_y,
            es_z=es_z,
        )
        loss = loss[0]
        expected_loss = np.array([[4067.5, 94.0, 103.6, 103.1, 4086.5]])

        npt.assert_allclose(loss, expected_loss, atol=1e-1)

    def test_get_diff_distances(self):
        """Test get_diff_distances for both 2D and 3D calculations."""
        es_x = np.array([10.0])
        es_y = np.array([15.0])
        es_z = np.array([19.0])
        imt_x = np.array([80.0, 50.0, 10.0, -80.0, 0.0])
        imt_y = np.array([15.0, 55.0, 95.0, 15.0, -40.0])
        imt_z = np.array([1.5, 3.0, 6.0, 7.5, 20.5])

        # 2D distances
        distances = self.propagation.get_diff_distances(
            imt_x,
            imt_y,
            imt_z,
            es_x,
            es_y,
            es_z,
            dist_2D=True,
        )
        expected_distances = (
            np.array([60.0, 35.4, 25.0, 60.0, 25.4]),
            np.array([10.0, 21.2, 55.0, 30.0, 30.5]),
        )
        npt.assert_allclose(distances, expected_distances, atol=1e-1)

        # 3D distances
        distances = self.propagation.get_diff_distances(
            imt_x,
            imt_y,
            imt_z,
            es_x,
            es_y,
            es_z,
        )
        expected_distances = (
            np.array([14.0, 9.0, 3.0, 6.7, -1.7]),
            np.array([60.0, 35.4, 25.0, 60.0, 25.4]),
            np.array([19.3, 25.9, 56.3, 31.8, 30.6]),
        )
        npt.assert_allclose(distances, expected_distances, atol=1e-1)

    def test_diffration_loss(self):
        """Test diffraction loss calculation for various scenarios."""
        # Test diffraction loss
        h = np.array([7.64, -0.56, -1.2, -0.1])
        d1 = np.array([34.99, 1060.15, 5.0, 120.0])
        d2 = np.array([25.02, 25.02, 2.33, 245.0])
        f = 40000 * np.ones_like(h)

        loss = self.propagation_diff_enabled.get_diffraction_loss(h, d1, d2, f)
        expected_loss = np.array([43.17, 0.0, 0.0, 4.48])

        npt.assert_allclose(loss, expected_loss, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
