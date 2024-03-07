import qem.preprocessing as pp
import pytest
import numpy as np
from math import pi

def test_foo_int():
    assert pp.foo(3, 4) == 7

def test_foo_float():
    assert pp.foo(3.14, pi) == pytest.approx(6.2815926535897)

def test_invert_image():
    assert np.all(pp.invert_image(np.array([[1, 2, 3], [4, 5, 6]])) == -np.array([[1, 2, 3], [4, 5, 6]]))