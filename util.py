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