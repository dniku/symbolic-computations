import sympy as sp
from tensor_product import TP
from transformer_applier import expand_with_transformers

from my_algebra import c, d, x, y, xy, yx, k
from my_resolution import Y, K, rho, psi, L, M, F, I, J, d3, d4


def dot(left, right):
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
            product[i][j] = expand_with_transformers(dot(row, col))
    return product


def validate_2k_2k1():
    left = [F, Y + K, rho, 0]
    right = [0, rho, Y + M, F]
    expr = dot(left, right)
    expr = expand_with_transformers(expr)
    assert expr == 0

    right = [0, 0, rho + J, Y]
    expr = dot(left, right)
    expr = expand_with_transformers(expr)
    assert expr == 0


def validate_2k1_2k2():
    left = [F, Y, rho, 0]
    right = [0, rho, Y + I, F]
    expr = dot(left, right)
    expr = expand_with_transformers(expr)
    assert expr == 0

    right = [0, 0, rho + J, Y]
    expr = dot(left, right)
    expr = expand_with_transformers(expr)
    assert expr == 0

if __name__ == '__main__':
    validate_2k_2k1()
    validate_2k1_2k2()

# comp = composition(d3, d4)
# for i in range(len(comp)):
#     row = comp[i]
#     for j in range(len(row)):
#         print('(%d, %d): %s' % (i, j, row[j]))
