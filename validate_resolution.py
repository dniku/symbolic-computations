import sympy as sp
from transformer_applier import texpand

from my_resolution import Y, rho, F, I, J, K, M


def dot(left, right):
    assert len(left) == len(right)
    return sp.Add(*[r * l for (l, r) in zip(left, right)])


def composition(d2, d1):
    """ Computes composition of d1 and d2, i.e. λx -> d2(d1(x)). """
    product_rows = len(d2)
    product_cols = len(d1[0])
    product = [
        [None] * product_cols
        for _ in range(product_rows)
    ]
    for i in range(product_rows):
        for j in range(product_cols):
            row = d2[i]
            col = [d1[k][j] for k in range(len(d1))]
            product[i][j] = texpand(dot(row, col))
    return product


def upper_left_2k_2k1():
    left = [
        [Y + I, rho + J, 0, 0],
        [F, Y + K, rho, 0],
    ]

    right = [
        [Y + M, rho + J, 0, 0],
        [F, Y, rho, 0],
        [0, 0, Y + M, rho + J],
        [0, 0, F, Y],
    ]

    return composition(left, right)


def upper_left_2k1_2k2():
    left = [
        [Y + M, rho + J, 0, 0],
        [F, Y, rho, 0],
    ]

    right = [
        [Y + I, rho + J, 0, 0],
        [F, Y + K, rho, 0],
        [0, 0, Y + I, rho + J],
        [0, 0, F, Y + K],
    ]

    return composition(left, right)
