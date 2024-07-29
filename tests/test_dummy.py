"""
Dummy test to test Github actions.
"""
import unittest


class TestDummy(unittest.TestCase):
    """
    A fake test that always returns true.
    """
    def test_dummy(self):
        """
        A test that always returns true.
        """
        # pylint: disable=redundant-unittest-assert
        self.assertTrue(True)
        # pylint: enable=redundant-unittest-assert
