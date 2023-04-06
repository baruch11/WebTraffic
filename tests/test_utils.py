import numpy as np
import unittest
from webtraffic.webtraffic_utils import smape_np


class TestSMAPE(unittest.TestCase):
    def test_smape_np(self):
        Mleft = np.array([1.0, 2.0, 3.0])
        Mright = np.array([2.0, 4.0, 6.0])
        expected_result = 66.66666666666667
        self.assertAlmostEqual(smape_np(Mleft, Mright), expected_result,
                               places=7)
