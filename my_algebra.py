import sympy as sp
from sympy import S

x, y = sp.symbols('x, y', commutative=False)
c, d = sp.symbols('c, d')

k, i = sp.symbols('k, i', integer=True)
xy = x * y
yx = y * x


# Not including x(yx)^k=0 because it's a Mul, not a Pow.
# I'm not handling those yet.
my_relations = {
    x**2: y * xy ** (k - 1) + c * xy ** k,
    y**2: d * xy ** k,
    yx**k: xy ** k
}

basis = [
    S.One,
    sp.sequence(y * xy**i, (i, 0, k - 1)),
    sp.sequence(x * yx**i, (i, 0, k - 1)),
    sp.sequence(xy**i, (i, 1, k - 1)),
    sp.sequence(yx**i, (i, 1, k - 1)),
    xy**k
]

zeros_mul = {
    '(x*y)**(k - 1)*y',
    'x*(x*y)**(k - 1)',
    'y*(y*x)**(k - 1)',
    'x**2*y',
    'x*(x*y)**k',
    '(x*y)**k*y',
    'y*(x*y)**k',
    '(y*x)**(k - 1)*x',
    # TODO: get rid at least of these
    '(x*y)**(k - 1)*(y*x)**(k - 1)',
    '(y*x)**(k - 1)*(x*y)**(k - 1)',
    '(x*y)**k*(y*x)**(k - 1)',
    '(y*x)**(k - 1)*(x*y)**k',
    'y*(x*y)**(2*k - 2)',
    'x*(y*x)**(2*k - 2)',
    'y*(x*y)**(k - 2)*y*(x*y)**(k - 1)'
}

zeros_pow = {
    '(x*y)**(k + 1)',
    '(y*x)**(k + 1)',
    '(x*y)**(2*k - 1)',
    '(y*x)**(2*k - 1)',
    '(x*y)**(2*k)',
}