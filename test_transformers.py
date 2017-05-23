import sympy as sp
from tensor_product import TP
from my_algebra import d, xy, k
from my_resolution import q
from transformers import Transformer_overlapping_sums_i_plus_1, Transformer_overlapping_sums_by_one


def main():
    pass


def test_sums():
    d0d1_00_1 = (
        d * sp.Sum(TP(xy ** q, xy ** (k - q)), (q, 0, k - 1)) +
        d * sp.Sum(TP(xy ** (q + 1), xy ** (k - q - 1)), (q, 0, k - 1)) +
        d * TP(1, xy ** k) + d * TP(xy ** k, 1)
    )
    d0d1_00_2 = (
        d * sp.Sum(TP(xy ** q, xy ** (k - q)), (q, 0, k - 1)) +
        d * sp.Sum(TP(xy ** q, xy ** (k - q)), (q, 1, k)) +
        d * TP(1, xy ** k) + d * TP(xy ** k, 1)
    )
    d0d1_00_3 = (
        d * sp.Sum(2 * TP(xy ** q, xy ** (k - q)), (q, 1, k - 1)) +
        2 * d * TP(1, xy ** k) + 2 * d * TP(xy ** k, 1)
    )
    assert Transformer_overlapping_sums_i_plus_1.match_transform(d0d1_00_1) == d0d1_00_2
    assert Transformer_overlapping_sums_by_one.match_transform(d0d1_00_2) == d0d1_00_3

if __name__ == '__main__':
    main()