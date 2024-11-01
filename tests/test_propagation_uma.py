# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:12:43 2017

@author: edgar
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.propagation.propagation_uma import PropagationUMa


class PropagationUMaTest(unittest.TestCase):

    def setUp(self):
        self.uma = PropagationUMa(np.random.RandomState())

    def test_los_probability(self):
        distance_2D = np.array([[10, 15, 40],
                                [17, 60, 80]])
        h_ue = np.array([1.5, 8, 15])
        los_probability = np.array([[1, 1, 0.74],
                                    [1, 0.57, 0.45]])
        npt.assert_allclose(self.uma.get_los_probability(distance_2D, h_ue),
                            los_probability,
                            atol=1e-2)

    def test_breakpoint_distance(self):
        h_bs = np.array([15, 20, 25, 30])
        h_ue = np.array([3, 4])
        h_e = np.ones((h_ue.size, h_bs.size))
        frequency = 30000 * np.ones(h_e.shape)
        breakpoint_distance = np.array([[11200, 15200, 19200, 23200],
                                        [16800, 22800, 28800, 34800]])
        npt.assert_array_equal(self.uma.get_breakpoint_distance(frequency, h_bs, h_ue, h_e),
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
        loss = np.array([[102.32, 115.99],
                         [108.09, 117.56],
                         [111.56, 118.90],
                         [114.05, 120.06]])
        npt.assert_allclose(self.uma.get_loss_los(distance_2D, distance_3D, frequency,
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
        loss = np.array([[62.32, 87.06],
                         [68.09, 84.39],
                         [71.56, 83.57],
                         [74.05, 83.40]])
        npt.assert_allclose(self.uma.get_loss_los(distance_2D, distance_3D, frequency,
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
        loss = np.array([[121.58, 148.29],
                         [132.25, 150.77],
                         [138.45, 152.78],
                         [142.70, 154.44]])
        npt.assert_allclose(self.uma.get_loss_nlos(distance_2D, distance_3D, frequency,
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
        loss = np.array([[120.02, 138.66],
                         [131.18, 149.83],
                         [146.13, 151.84],
                         [141.75, 153.51]])
        npt.assert_allclose(self.uma.get_loss_nlos(distance_2D, distance_3D, frequency,
                                                   h_bs, h_ue, h_e, shadowing_std),
                            loss,
                            atol=1e-2)


if __name__ == '__main__':
    unittest.main()
