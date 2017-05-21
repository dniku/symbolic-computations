from collections import deque
from typing import List

import sympy as sp
from sympy import S

from tensor_product import TP
from transformers import Transformer, transformers_general, transformers_mul
from my_algebra import zeros_mul, zeros_pow
from my_transformers import transformers_relations_general, transformers_relations_mul


config = {
    'char': 2,
    'general': transformers_general + transformers_relations_general,
    'mul': transformers_mul + transformers_relations_mul
}


def apply_transformer_termwise(terms: List, tr: Transformer) -> List:
    # TODO: get rid of deque (replace with a custom class based on list)
    queue = deque(terms)
    terms = []

    while len(queue) >= tr.arg_num:
        current_terms = []
        for _ in range(tr.arg_num):
            current_terms.append(queue.popleft())

        if tr.match(*current_terms):
            new_terms = tr.transform(*current_terms)
            while new_terms:
                queue.appendleft(new_terms.pop())
        else:
            while len(current_terms) > 1:
                queue.appendleft(current_terms.pop())
            terms.append(current_terms.pop())
            assert not current_terms
    while queue:
        terms.append(queue.popleft())

    return terms


def try_transformers_general(expr):
    for tr in config['general']:
        assert tr.arg_num == 1
        assert tr.context == 'none'  # ?

        if tr.match(expr):
            return tr.transform(expr)

    return expr


def apply_characteristic(x):
    return x if config['char'] == 0 else x % config['char']


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


def expand_mul(expr):
    assert isinstance(expr, sp.Mul)

    commutative, other = split_commutative(expr)

    commutative = [(apply_characteristic(v) if isinstance(v, sp.Number) else v) for v in commutative]
    commutative = sp.Mul(*commutative)
    if commutative == 0:
        return S.Zero

    terms = [expand_with_transformers_impl(term) for term in other]

    for tr in config['mul']:
        terms = apply_transformer_termwise(terms, tr)

    def catch_zeros(expr):
        s = str(expr)
        for z in zeros_mul:
            if z in s:
                return S.Zero
        for z in zeros_pow:
            if z in s:
                return S.Zero
        return expr

    def postprocess(expr):
        # TODO: are these actually needed?
        # expr = sp.powsimp(expr)
        # expr = sp.expand_mul(expr)
        # TODO: get rid of this
        expr = catch_zeros(expr)
        return expr

    # FIXME: THIS COULD BE A BUG if one of the transformers yielded something
    # like an Add whose part could be matched by catch_zeros and thus the whole
    # expression would be marked as zero, which it is not. Currently there are no
    # such transformers in mul, but there may be.
    # Anyway, catch_zeros is evil and should be abandoned.
    return commutative * postprocess(sp.Mul(*terms))


def expand_pow(expr):
    assert isinstance(expr, sp.Pow)

    base = expand_with_transformers_impl(expr.args[0])
    expr = sp.Pow(base, expr.args[1])
    expr = try_transformers_general(expr)

    return expr


def expand_sum(expr):
    function = expand_with_transformers_impl(expr.function)
    if function == 0:  # actually, this is an inlined transformer
        return S.Zero
    elif isinstance(function, sp.Add):
        terms = [sp.Sum(arg, expr.limits) for arg in function.args]
        return sp.Add(*terms)
    elif isinstance(function, sp.Mul):
        # At this point, `function` is already expanded, hence no need to call expand_mul.
        # So what's left to do is only to split away the commutative part.
        fc, fv = split_commutative(function)  # `fv` is a list of terms
        commutative = sp.Mul(*fc)
        expr = sp.Sum(sp.Mul(*fv), expr.limits[0])
        expr = try_transformers_general(expr)
        return commutative * expr

    return sp.Sum(function, expr.limits)


def expand_tp(expr):
    l, r = expr.args
    l = expand_with_transformers_impl(l)
    r = expand_with_transformers_impl(r)
    if l == 0 or r == 0:
        return S.Zero

    if isinstance(l, sp.Add):
        expr = sp.Add(*[TP(arg, r) for arg in l.args])
        return expand_with_transformers_impl(expr)
    if isinstance(r, sp.Add):
        expr = sp.Add(*[TP(l, arg) for arg in r.args])
        return expand_with_transformers_impl(expr)

    lc, lnc = split_commutative(l) if isinstance(l, sp.Mul) else ([S.One], [l])
    rc, rnc = split_commutative(r) if isinstance(r, sp.Mul) else ([S.One], [r])

    lnc = sp.Mul(*lnc)
    rnc = sp.Mul(*rnc)

    commutative = sp.Mul(*(lc + rc))
    tp = TP(lnc, rnc)

    return commutative * tp


def expand_seq(expr):
    formula = expand_with_transformers_impl(expr.formula)
    formula = sp.piecewise_fold(formula)
    expr = sp.SeqFormula(formula, expr.args[1])
    return try_transformers_general(expr)


def expand_piecewise(expr):
    def expand_cond_expr(f, cond):
        return (expand_with_transformers_impl(f), cond)
    args = [expand_cond_expr(*arg) for arg in expr.args]
    expr = sp.Piecewise(*args)
    return try_transformers_general(expr)


def expand_with_transformers_impl(expr):
    # because, well, expand() doesn't work on SeqFormula
    # https://github.com/sympy/sympy/issues/12634
    if not isinstance(expr, sp.SeqFormula):
        # this also does sympification
        expr = sp.expand(expr, basic=True, mul=True, multinomial=True,
                         power_base=False, power_exp=False, log=False)
    expr = sp.powsimp(expr)

    if isinstance(expr, sp.Number):
        return apply_characteristic(expr)
    elif isinstance(expr, sp.Symbol):
        return try_transformers_general(expr)
    elif isinstance(expr, sp.Add):
        # FIXME: this assumes that no relations have a sum
        # FIXME: asymmetry, other cases have dedicated expand_* functions
        terms = [expand_with_transformers_impl(term) for term in expr.args]
        return sp.Add(*terms)
    elif isinstance(expr, sp.Mul):
        return expand_mul(expr)
    elif isinstance(expr, sp.Pow):
        return expand_pow(expr)
    elif isinstance(expr, sp.Sum):
        return expand_sum(expr)
    elif isinstance(expr, TP):
        return expand_tp(expr)
    elif isinstance(expr, sp.SeqFormula):
        return expand_seq(expr)
    elif isinstance(expr, sp.Piecewise):
        return expand_piecewise(expr)
    else:
        raise TypeError("%s is unmatched type %s" % (expr, type(expr)))


def test_issue_11981():
    x, y = sp.symbols('x y', commutative=False)
    assert sp.powsimp((x*y)**2 * (y*x)**2) == (x*y)**2 * (y*x)**2


def expand_with_transformers(expr, max_iterations=10):
    """ Apply transformers to an arbitrary Sympy expression while possible. """

    # because otherwise something is bound to break in a most unexpected way
    test_issue_11981()

    for _ in range(max_iterations):
        new_expr = expand_with_transformers_impl(expr)
        if expr == new_expr:
            return expr
        expr = new_expr

    raise RuntimeError("No convergence after %d iterations" % max_iterations)


texpand = expand_with_transformers
