import pvml
import pytest


@pytest.mark.parametrize("name, classes",
                         [("gaussian", 2), ("gaussian4", 4), ("xor", 2),
                          ("rings", 2), ("moons", 2), ("swissroll", 2),
                          ("categorical", 2), ("yinyang", 2), ("iris", 3)])
def test_load_dataset(name, classes):
    X, Y = pvml.load_dataset(name)
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 2
    assert Y.max() == classes - 1
