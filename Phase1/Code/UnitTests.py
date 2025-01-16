import Wrapper as wrap
import unittest
import numpy as np

class TestConvolveFunction(unittest.TestCase):
    def test_Convolve2D(self):
        input_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        kernel = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        expected_output = np.array([
            [-13.0, -20.0, -17.0],
            [-18.0, -24.0, -18.0],
            [13.0, 20.0, 17.0]
        ])
        result = wrap.convolve2D(input_matrix, kernel)
        np.testing.assert_array_equal(result, expected_output)

if __name__ == '__main__':
    unittest.main()
