import sympy as sp
from sympy import S


def sum_terms(terms):
    if not terms:
        return S.Zero
    elif isinstance(terms[0], sp.SeqFormula):
        acc = terms[0]
        for item in terms[1:]:
            acc += item
        return acc
    else:
        return sp.Add(*terms)


def split_commutative(expr):
    assert isinstance(expr, sp.Mul)
    # TODO: make sure there are no cases where this is needed
    # if not isinstance(expr, sp.Mul):
    #     return S.One, [expr]

    commutative = []
    other = []
    for arg in expr.args:
        assert not isinstance(arg, int)
        if isinstance(arg, sp.Number):
            # Making sure Sympy makes sense
            assert arg.is_commutative
        if arg.is_commutative:
            commutative.append(arg)
        else:
            other.append(arg)

    return commutative, other