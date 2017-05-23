import sympy as sp

from my_algebra import basis
from tensor_product import TP
from util import sum_terms, split_commutative
from transformer_applier import texpand


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
    elif isinstance(expr, sp.Mul):
        commutative, other = split_commutative(expr)
        commutative = sp.Mul(*commutative)
        other = sp.Mul(*other)
        if commutative == 1:
            raise TypeError("Don't know how to handle this product: {}".format(expr))
        else:
            result = ast(other, r)
            return multiply_termwise(commutative, result)
    elif isinstance(expr, sp.Sum):
        new_function = ast(expr.function, r)
        if isinstance(new_function, sp.SeqFormula):
            return sp.SeqFormula(sp.Sum(new_function.formula, expr.limits[0]), new_function.args[1])
        else:
            return sp.Sum(new_function, expr.limits[0])
    else:
        raise TypeError("Unknown type {} for expression {}".format(type(expr), expr))

def ast_table(expr):
    expr_expanded = texpand(expr)
    for r in basis:
        product = ast(expr_expanded, r)
        product = texpand(product)
        print('({}) ∗ {}\n    = {}'.format(expr, r, product))


def ast_matrix(expr):
    expr_expanded = texpand(expr)
    col1 = basis
    col2 = [texpand(ast(expr_expanded, r)) for r in basis]
    return sp.Matrix([col1, col2]).T
