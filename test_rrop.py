import sympy as sp

from rrop import TP, Sum, mysimplify, k, c, d, x, y, xy, yx, canonical_expr


def main():
    sp.init_printing(wrap_line=False)

    i = sp.symbols('i')
    X = TP(x, 1) + TP(1, x)

    v = x * x * y * x
    print(canonical_expr(v))

    return

    Ym = TP(y, 1) - TP(1, y)
    Yp = TP(y, 1) + TP(1, y)

    d011 = Ym * Ym + Xm * 0
    print(mysimplify(d011))

    print(TP(x ** 2, x))

    print(TP(y * xy ** (k - 1 - i) * x, 1))
    print(TP(x, 1) * TP(1, x) + Sum(TP((x * y) ** (k - 1) * y, x), (i, 0, k - 1)))

    print(TP(x, 0))
    print(TP(0, y))

    Phi1 = sp.Sum(TP((y * x) ** i, (x * y) ** (k - 1 - i)), (i, 0, k - 1))
    Phi2 = -TP(x, 1) - TP(1, x) + sp.Sum(TP(y * (x * y) ** i, y * (x * y) ** (k - 2 - i)), (i, 0, k - 2))
    d012 = Ym * Phi1 + Xm * Phi2

    print(d012)
    print(mysimplify(d012))


def test_zero_proparagion():
    assert TP(x, 0) == 0
    assert TP(0, y) == 0
    assert TP(xy**k, y) * TP(x, 1) == 0
    assert TP(x * c**2 * d * xy**k, y) == 0


def test_sign_removal():
    X = TP(x, 1) - TP(1, x)
    assert X == TP(x, 1) + TP(1, x)


def test_simplification():
    assert canonical_expr(y * x * y * x) == yx ** 2
    assert canonical_expr(d * xy ** (k + 1)) == 0
    assert canonical_expr(x * y * xy ** (k - 1)) == xy ** k
    assert canonical_expr(y * xy ** (k - 1) * yx ** (k - 1)) == 0

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

if __name__ == '__main__':
    main()
