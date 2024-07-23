"""
Dummy test for the dummy module.
"""
import unittest

from src.prsdk.dummy import compute_percent_change

class TestDummy(unittest.TestCase):
    """
    Tests for the dummy module.
    """
    def test_pct_change(self):
        """
        Tests the compute_percent_change function.
        It should return the input divided by 100.
        """
        self.assertEqual(compute_percent_change(100), 1.0)
        self.assertEqual(compute_percent_change(50), 0.5)
        self.assertEqual(compute_percent_change(0), 0.0)
