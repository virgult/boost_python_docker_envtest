import EnvTest
import unittest
import numpy as np

class TestEnvTest(unittest.TestCase):

    def test_print(self):
        EnvTest.EnvTest.TestPrint()

    def test_sum_two_channels(self):
        _ = EnvTest.EnvTest(1.)
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).astype(np.float32)
        y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).astype(np.float32)
        z = np.zeros(x.shape).astype(np.float32)
        EnvTest.EnvTest.TestSum(x, y, z)
        for n in range(x.shape[0]):
            for m in range(x.shape[1]):
                self.assertEqual(z[n][m], x[n][m] + y[n][m])

    def test_sum_single_channel(self):
        _ = EnvTest.EnvTest(1.)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        z = np.zeros(x.shape).astype(np.float32)
        EnvTest.EnvTest.TestSum(x, y, z)
        for n in range(x.shape[0]):
            self.assertEqual(z[n], x[n] + y[n])

    def test_gain_two_channels(self):
        e = EnvTest.EnvTest(-5.)
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).astype(np.float32)
        y = np.zeros(x.shape).astype(np.float32)
        e.TestGain(x, y)
        for n in range(x.shape[0]):
            for m in range(x.shape[1]):
                self.assertEqual(y[n][m], x[n][m] * -5.)

    def test_gain_single_channel(self):
        e = EnvTest.EnvTest(-5.)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.zeros(x.shape).astype(np.float32)
        e.TestGain(x, y)
        for n in range(x.shape[0]):
            self.assertEqual(y[n], x[n] * -5.)


if __name__ == '__main__':
    unittest.main()
