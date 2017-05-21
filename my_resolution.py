import sympy as sp
from my_algebra import x, y, xy, yx, k, c, d
from tensor_product import TP

q = sp.symbols('q')

Y = TP(y, 1) + TP(1, y)
X = TP(x, 1) + TP(1, x)

phi1 = sp.Sum(TP(yx**q, xy**(k-1-q)), (q, 0, k - 1))
phi2 = X + sp.Sum(TP(y*xy**q, y*xy**(k-2-q)), (q, 0, k - 2))

rho = TP(x*yx**(k-1), 1) + TP(1, x*yx**(k-1))
psi = TP(yx**(k-1), xy**(k-1))
tau = Y * X
sigma = X * phi1

lbda = TP(x, 1) * phi1
mu = sp.Sum(TP((x*y)**q, y*(x*y)**(k - 1 - q)), (q, 0, k - 1))
omega = TP((1 + c*x) * yx**(k-1), xy**k)

A = c**2 * d * omega
B = d * TP(1 + c*x, y) * X
C = (
    + c**2 * TP(y*xy**(k-1), x*yx**(k-1))
    + c**3 * TP(x*yx**(k-1), xy**k)
)
D = (
    + c * Y * TP(x, x)
    + c * TP(x**2, y)
    + d * TP(x*yx**(k-1), x)
)

E = (
    + d * TP(x*yx**(k-1), 1)
    + c**2 * d * omega
)
F = d * TP(1, y)
G = c**2 * omega
H = c * TP(y*xy**(k-1), xy**(k-1))

I = c**3 * d * TP(x*yx**(k-1), xy**k)
J = c**3 * TP(x*yx**(k-1), xy**k)
K = d * TP(x*yx**(k-1), 1)
L = c * TP(yx**(k-1), x*yx**(k-1))

M = d * TP(x*yx**(k-1), 1) + c**3 * d * TP(x*yx**(k-1), xy**k)

d0 = [
    [Y, X],
]

d1 = [
    [Y + d * lbda, phi1 + c * lbda],
    [d * mu, phi2 + c * mu],
]

d2 = [
    [Y + A, rho + C],
    [B, tau + D],
]

d3 = [
    [Y + E, rho + G, 0],
    [F, Y, sigma + H],
]

d4 = [
    [Y + I, rho + J, 0, 0],
    [F, Y + K, rho, psi + L],
    [0, 0, Y, X],
]

d5 = [
    [Y + M, rho + J, 0, 0],
    [F, Y, rho, 0],
    [0, 0, Y + d * lbda, phi1 + c * lbda],
    [0, 0, d * mu, phi2 + c * mu],
]