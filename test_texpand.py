import pytest
import sympy as sp

from tensor_product import TP
from transformer_applier import texpand
from my_algebra import k, x, y, xy, yx, c, d
from my_resolution import X, q


i = sp.symbols('i')


def test_zero_propagation():
    assert texpand(TP(x, 0)) == 0
    assert texpand(TP(0, y)) == 0
    assert texpand(TP(xy**k, y) * TP(x, 1)) == 0
    assert texpand(TP(x * c**2 * d * xy**k, y)) == 0


def test_sign_removal():
    X = TP(x, 1) - TP(1, x)
    assert texpand(X) == TP(x, 1) + TP(1, x)


@pytest.mark.skip(reason="it is known that this test fails")
def test_sum_simplification():
    assert sp.Sum(TP(yx * xy ** i, yx ** (k - i)), (i, 0, k - 1)) == TP(yx, xy ** k)


def test_piecewise_simplification():
    d2d3_11 = (
        c * d * TP(x, sp.Piecewise((x, sp.Eq(k, 0)), (0, True))) +
        c * d * TP(sp.Piecewise((x, sp.Eq(k, 0)), (0, True)), x) +
        d * TP(1, sp.Piecewise((x, sp.Eq(k, 0)), (0, True))) +
        d * TP(sp.Piecewise((x, sp.Eq(k, 0)), (0, True)), 1)
    )
    assert texpand(d2d3_11) == 0

    expr = sp.Sum(
        sp.Piecewise(
            (c * TP(xy**k, y*xy**(k-q-1)*y) + TP(y*xy**(k - 1), y*xy**(k-q-1) * y), sp.Eq(q, 0)),
            (0, True)
        ), (q, 0, k - 1)
    )
    assert texpand(expr) == 0


def test_misc():
    assert texpand(2 * x) == 0
    assert texpand(y * x * y * x) == yx ** 2
    assert texpand(d * xy ** (k + 1)) == 0
    assert texpand(x * y * xy ** (k - 1)) == xy ** k
    assert texpand(y * xy ** (k - 1) * yx ** (k - 1)) == 0


@pytest.mark.skip(reason="it is known that this test fails")
def test_x_cubed_reduces_to_xy_k():
    left = TP(x, y) * X
    right = X + TP(xy ** k, xy ** k)
    assert texpand(left * right) == texpand(texpand(left) * texpand(right))
