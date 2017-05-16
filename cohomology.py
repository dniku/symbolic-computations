import sympy as sp

from my_algebra import basis
from tensor_product import TP
from util import sum_terms
from transformer_applier import expand_with_transformers, split_commutative


def multiply_termwise(coeff, expr):
    if isinstance(expr, sp.SeqFormula):
        return sp.SeqFormula(coeff * expr.formula, expr.args[1])
    else:
        return coeff * expr


def ast(expr, r):
    if isinstance(expr, TP):
        if isinstance(r, sp.SeqFormula):
            formula = ast(expr, r.formula)
            limits = r.args[1]
            return sp.SeqFormula(formula, limits)
        else:
            return expr.args[0] * r * expr.args[1]
    elif isinstance(expr, sp.Add):
        terms = [ast(term, r) for term in expr.args]
        return sum_terms(terms)
    else:
        commutative, other = split_commutative(expr)
        commutative = sp.Mul(*commutative)
        other = sp.Mul(*other)
        if commutative == 1:
            raise TypeError("Unknown type {} for expression {}".format(type(expr), expr))
        else:
            result = ast(other, r)
            return multiply_termwise(commutative, result)


def ast_table(expr):
    for r in basis:
        product = ast(expr, r)
        product_expanded = expand_with_transformers(product)
        print('({}) ∗ {}\n    = {}\n    → {}'.format(expr, r, product, product_expanded))
        # if product != product_expanded:
        #     print('%s → %s' % (product, product_expanded))
        # else:
        #     print(product_expanded)

# from my_resolution import tau as expr
# print(expr)
# expr = expand_with_transformers(expr)
# print(expr)
# ast_table(expr)
