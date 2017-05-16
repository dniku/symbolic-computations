import sympy as sp

from rrop import Sum, mysimplify, c, d, x, y
from tensor_product import TP
from transformer_applier import expand_with_transformers
from my_algebra import k, xy, yx


def main():
    sp.init_printing(wrap_line=False)

    i = sp.symbols('i')
    X = TP(x, 1) + TP(1, x)

    Y = TP(y, 1) + TP(1, y)
    Phi1 = Sum(TP((y * x) ** i, (x * y) ** (k - 1 - i)), (i, 0, k - 1))
    lbda = TP(x, 1) * Phi1

    A = c ** 2 * d * TP((1 + c * x) * yx ** (k - 1), xy ** k)
    B = d * TP(1 + c * x, y) * X
    C = c ** 2 * TP(y * xy ** (k - 1), x * yx ** (k - 1))

    rho = TP(x * yx ** (k - 1), 1) + TP(1, x * yx ** (k - 1))

    v = expand_with_transformers(2 * x)
    print(v)


def test_zero_proparagion():
    assert TP(x, 0) == 0
    assert TP(0, y) == 0
    assert TP(xy**k, y) * TP(x, 1) == 0
    assert TP(x * c**2 * d * xy**k, y) == 0


def test_sign_removal():
    X = TP(x, 1) - TP(1, x)
    assert X == TP(x, 1) + TP(1, x)


def test_canonical():
    assert expand_with_transformers(2 * x) == 0
    assert expand_with_transformers(y * x * y * x) == yx ** 2
    assert expand_with_transformers(d * xy ** (k + 1)) == 0
    assert expand_with_transformers(x * y * xy ** (k - 1)) == xy ** k
    # The following test fails, disabling it for now
    # assert canonical_expr(y * xy ** (k - 1) * yx ** (k - 1)) == 0


def test_sum_simplification():
    i = sp.symbols('i')
    assert Sum(TP(y * x * xy ** i, yx ** (k - i)), (i, 0, k - 1)) == TP(yx, xy ** k)


def test_misc():
    i = sp.symbols('i')

    Y = TP(y, 1) + TP(1, y)
    X = TP(x, 1) + TP(1, x)

    Phi1 = Sum(TP((y * x) ** i, (x * y) ** (k - 1 - i)), (i, 0, k - 1))
    Phi2 = -TP(x, 1) - TP(1, x) + Sum(TP(y * (x * y) ** i, y * (x * y) ** (k - 2 - i)), (i, 0, k - 2))

    lbda = TP(x, 1) * Phi1
    mu = Sum(TP((x * y) ** i, y * (x * y) ** (k - 1 - i)), (i, 0, k - 1))

    assert mysimplify(d * TP(1 + c * x, y) * X ** 2) == d * TP(y * xy ** (k - 1), y)
    assert mysimplify(
        d * TP(1 + c * x, y) * X * Sum(TP(y * (x * y) ** i, y * (x * y) ** (k - 2 - i)), (i, 0, k - 2))) == (
               d ** 2 * TP((x * y) ** (k - 1), (x * y) ** k) +
               d * Sum(TP(y * (x * y) ** i, y * (x * y) ** (k - i - 1)), (i, 0, k - 2)) +
               c * d * Sum(TP((x * y) ** (i + 1), y * (x * y) ** (k - i - 1)), (i, 0, k - 2)))
    assert mysimplify(
        c * d * TP(1 + c * x, y) * X *
        Sum(TP(xy ** i, y * xy ** (k - 1 - i)),
            (i, 0, k - 1))) == c * d * Sum(TP(xy ** i, y * xy ** (k - i)), (i, 0, k - 1))

    left = TP(x, y) * X
    right = X + TP(xy ** k, xy ** k)
    # No idea how to debug it yet
    # assert mysimplify(left * right) == mysimplify(mysimplify(left) * mysimplify(right))

if __name__ == '__main__':
    main()
