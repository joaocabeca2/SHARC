
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:14:56 2017

@author: LeticiaValle_Mac
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.propagation.propagation_umi import PropagationUMi


class PropagationUMiTest(unittest.TestCase):

    def setUp(self):
        los_adjustment_factor = 18
        self.umi = PropagationUMi(
            np.random.RandomState(),
            los_adjustment_factor)

    def test_los_probability(self):
        distance_2D = np.array([[10, 15, 40],
                                [17, 60, 80]])
        los_probability = np.array([[1, 1, 0.631],
                                    [1, 0.432, 0.308]])
        npt.assert_allclose(self.umi.get_los_probability(distance_2D,
                                                         self.umi.los_adjustment_factor),
                            los_probability,
                            atol=1e-2)

    def test_breakpoint_distance(self):
        h_bs = np.array([15, 20, 25, 30])
        h_ue = np.array([3, 4])
        h_e = np.ones((h_ue.size, h_bs.size))
        frequency = 30000 * np.ones(h_e.shape)
        breakpoint_distance = np.array([[11200, 15200, 19200, 23200],
                                        [16800, 22800, 28800, 34800]])
        npt.assert_array_equal(self.umi.get_breakpoint_distance(frequency, h_bs, h_ue, h_e),
                               breakpoint_distance)

    def test_loss_los(self):
        distance_2D = np.array([[100, 500],
                                [200, 600],
                                [300, 700],
                                [400, 800]])
        h_bs = np.array([30, 35])
        h_ue = np.array([2, 3, 4, 5])
        h_e = np.ones(distance_2D.shape)
        distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[:, np.newaxis])**2)
        frequency = 30000 * np.ones(distance_2D.shape)
        shadowing_std = 0
        loss = np.array([[104.336, 118.690],
                         [110.396, 120.346],
                         [114.046, 121.748],
                         [116.653, 122.963]])
        npt.assert_allclose(self.umi.get_loss_los(distance_2D, distance_3D, frequency,
                                                  h_bs, h_ue, h_e, shadowing_std),
                            loss,
                            atol=1e-2)

        distance_2D = np.array([[100, 500],
                                [200, 600],
                                [300, 700],
                                [400, 800]])
        h_bs = np.array([30, 35])
        h_ue = np.array([2, 3, 4, 5])
        h_e = np.ones(distance_2D.shape)
        distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[:, np.newaxis])**2)
        frequency = 300 * np.ones(distance_2D.shape)
        shadowing_std = 0
        loss = np.array([[64.336, 89.215],
                         [70.396, 86.829],
                         [74.046, 86.187],
                         [76.653, 86.139]])
        npt.assert_allclose(self.umi.get_loss_los(distance_2D, distance_3D, frequency,
                                                  h_bs, h_ue, h_e, shadowing_std),
                            loss,
                            atol=1e-2)

    def test_loss_nlos(self):
        distance_2D = np.array([[100, 500],
                                [200, 600],
                                [300, 700],
                                [400, 800]])
        h_bs = np.array([30, 35])
        h_ue = np.array([2, 3, 4, 5])
        h_e = np.ones(distance_2D.shape)
        distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[:, np.newaxis])**2)
        frequency = 30000 * np.ones(distance_2D.shape)
        shadowing_std = 0
        loss = np.array([[128.84, 152.96],
                         [138.72, 155.45],
                         [144.56, 157.50],
                         [148.64, 159.25]])
        npt.assert_allclose(self.umi.get_loss_nlos(distance_2D, distance_3D, frequency,
                                                   h_bs, h_ue, h_e, shadowing_std),
                            loss,
                            atol=1e-2)

        distance_2D = np.array([[1000, 3000],
                                [2000, 6000],
                                [5000, 7000],
                                [4000, 8000]])
        h_bs = np.array([30, 35])
        h_ue = np.array([2, 3, 4, 5])
        h_e = np.ones(distance_2D.shape)
        distance_3D = np.sqrt(distance_2D**2 + (h_bs - h_ue[:, np.newaxis])**2)
        frequency = 300 * np.ones(distance_2D.shape)
        shadowing_std = 0
        loss = np.array([[120.96, 137.80],
                         [131.29, 148.13],
                         [145.03, 150.19],
                         [141.31, 151.94]])
        npt.assert_allclose(self.umi.get_loss_nlos(distance_2D, distance_3D, frequency,
                                                   h_bs, h_ue, h_e, shadowing_std),
                            loss,
                            atol=1e-2)


if __name__ == '__main__':
    unittest.main()
