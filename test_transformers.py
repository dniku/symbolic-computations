import sympy as sp
from tensor_product import TP
from my_algebra import d, x, y, xy, k, c
from my_resolution import q
from transformers import Transformer_overlapping_sums_i_plus_1, Transformer_overlapping_sums_by_one, Transformer_sum_of_piecewise_with_only_zero
from my_transformers import Transformer_k_neq_0_in_piecewise


def test_sums():
    d0d1_00_1 = (
        d * sp.Sum(TP(xy**q, xy**(k - q)), (q, 0, k-1)) +
        d * sp.Sum(TP(xy**(q + 1), xy**(k-q-1)), (q, 0, k-1)) +
        d * TP(1, xy**k) + d * TP(xy**k, 1)
    )
    d0d1_00_2 = (
        d * sp.Sum(TP(xy**q, xy**(k - q)), (q, 0, k-1)) +
        d * sp.Sum(TP(xy**q, xy**(k - q)), (q, 1, k)) +
        d * TP(1, xy**k) + d * TP(xy**k, 1)
    )
    d0d1_00_3 = (
        d * sp.Sum(2 * TP(xy**q, xy**(k - q)), (q, 1, k-1)) +
        2 * d * TP(1, xy**k) + 2 * d * TP(xy**k, 1)
    )
    assert Transformer_overlapping_sums_i_plus_1.match_transform(d0d1_00_1) == d0d1_00_2
    assert Transformer_overlapping_sums_by_one.match_transform(d0d1_00_2) == d0d1_00_3


def test_piecewise():
    d2d3_11 = (
        sp.Piecewise((x, sp.Eq(k, 0)), (0, True))
    )
    assert Transformer_k_neq_0_in_piecewise.match_transform(d2d3_11) == 0


def test_piecewise_within_sum():
    expr_1 = sp.Sum(
        sp.Piecewise(
            (
            c * TP(xy**k, y*xy**(k-q-1)*y) + TP(y*xy**(k - 1), y*xy**(k-q-1) * y), sp.Eq(q, 0)),
            (0, True)
        ), (q, 0, k - 1)
    )
    expr_2 = c * TP(xy**k, y*xy**(k-1)*y) + TP(y*xy**(k - 1), y*xy**(k-1) * y)
    assert Transformer_sum_of_piecewise_with_only_zero.match_transform(expr_1) == expr_2
