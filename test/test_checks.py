import numpy as np
import pvml
import pvml.checks
import unittest


class TestLogisticRegression(unittest.TestCase):
    def test_check_size_ok(self):
        pvml.checks._check_size("mn, nm, mm", np.empty((3, 5)), np.empty((5, 3)), np.empty((3, 3)))

    def test_check_optional(self):
        pvml.checks._check_size("mn?, mn?, mn?", None, np.empty((2, 4)), None)

    def test_check_arg_error(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_size("mn, nm", np.empty(3,))

    def test_check_dim_error(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_size("mn", np.empty(3,))

    def test_check_size_error(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_size("mn, nm", np.empty((2, 4)), np.empty((2, 4)))

    def test_check_scalar_ok(self):
        pvml.checks._check_size("*", 42)

    def test_check_scalar_error(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_size("*", np.array([42]))

    def test_check_labels_ok(self):
        pvml.checks._check_labels((np.arange(10) % 4).astype(float), 4)

    def test_check_labels_error1(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_labels((np.arange(10) % 4) + 0.5)

    def test_check_labels_error2(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_labels(np.arange(5, -5, -1))

    def test_check_labels_error3(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_labels(np.arange(10) % 4, 3)

    def test_check_categorical_ok(self):
        x = (np.arange(10).reshape(5, 2) % 4).astype(float)
        pvml.checks._check_categorical(x)

    def test_check_categorical_error1(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_categorical((np.arange(12).reshape(4, 3) % 4) + 0.5)

    def test_check_categorical_error2(self):
        with self.assertRaises(ValueError):
            pvml.checks._check_categorical(np.arange(5, -5, -1))


if __name__ == '__main__':
    unittest.main()
