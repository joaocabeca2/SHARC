import unittest
import numpy as np
import numpy.testing as npt
import satellite.utils.sat_utils as sat_utils


class TestSatUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_ecef2lla(self):
        # Object is over the meridional plane at 1414km of altitude
        sx = 7792137.0
        sy = 0.0
        sz = 0.0
        lat, lng, alt = sat_utils.ecef2lla(sx, sy, sz)
        npt.assert_almost_equal(lat, 0.0, 2)
        npt.assert_almost_equal(lng, 0.0, 2)
        npt.assert_almost_equal(alt, 1414000.0, 1)

        # Object is over the meridional plane at sea level
        sx = 6378137.0
        sy = 0.0
        sz = 0.0
        lat, lng, alt = sat_utils.ecef2lla(sx, sy, sz)
        npt.assert_almost_equal(lat, 0.0, 2)
        npt.assert_almost_equal(lng, 0.0, 2)
        npt.assert_almost_equal(alt, 0.0, 1)

        sx = 7792137.0
        sy = 6378137.0
        sz = 3264751.4
        lat, lng, alt = sat_utils.ecef2lla(sx, sy, sz)
        npt.assert_almost_equal(lat, 18.0316, 3)
        npt.assert_almost_equal(lng, 39.30153, 3)
        npt.assert_almost_equal(alt, 4209582, 1)

        # test the array form
        sx = [7792137.0, 6378137.0, 7792137.0]
        sy = [0.0, 0.0, 6378137.0]
        sz = [0.0, 0.0, 3264751.4]
        lat, lng, alt = sat_utils.ecef2lla(sx, sy, sz)
        npt.assert_almost_equal(lat, [0.0, 0.0, 18.0316], 3)
        npt.assert_almost_equal(lng, [0.0, 0.0, 39.30153], 3)
        npt.assert_almost_equal(alt, [1414000.0, 0.0, 4209582], 1)

    def test_lla2ecef(self):
        # Object is over the meridional plane at 1414km of altitude
        lat = 0.0
        lng = 0.0
        alt = 1414000.0
        sx, sy, sz = sat_utils.lla2ecef(lat, lng, alt)
        npt.assert_almost_equal(sx, 7792137.0, 1)
        npt.assert_almost_equal(sy, 0.0, 1)
        npt.assert_almost_equal(sz, 0.0, 1)

        # Object is over the meridional plane at sea level
        lat = 0.0
        lng = 0.0
        alt = 0.0
        sx, sy, sz = sat_utils.lla2ecef(lat, lng, alt)
        npt.assert_almost_equal(sx, 6378137.0, 1)
        npt.assert_almost_equal(sy, 0.0, 1)
        npt.assert_almost_equal(sz, 0.0, 1)

        lat = 43.0344
        lng = 46.839308
        alt = 141400.0
        sx, sy, sz = sat_utils.lla2ecef(lat, lng, alt)
        npt.assert_almost_equal(sx, 3264751.4, 1)
        npt.assert_almost_equal(sy, 3481390.4, 1)
        npt.assert_almost_equal(sz, 4426792.5, 1)

        # test the array form
        lat = [0.0, 0.0, 43.0344]
        lng = [0.0, 0.0, 46.839308]
        alt = [1414000.0, 0.0, 141400.0]
        sx, sy, sz = sat_utils.lla2ecef(lat, lng, alt)
        npt.assert_almost_equal(sx, [7792137.0, 6378137.0, 3264751.4], 1)
        npt.assert_almost_equal(sy, [0.0, 0.0, 3481390.4], 1)
        npt.assert_almost_equal(sz, [0.0, 0.0, 4426792.5], 1)


if __name__ == '__main__':
    unittest.main()
