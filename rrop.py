# http://docs.sympy.org/dev/guide.html#sympy-s-architecture

import sys
from collections import deque

from sympy.core.singleton import S

from util import split_commutative
from constants import WARNINGS
from matchers import *
from tensor_product import TP
from transformers import Transformer_tp_tp, Transformer_sum_tp, Transformer_tp_sum


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def debug_warning(s):
    if WARNINGS:
        eprint('WARNING:', s)


def expand_tp_mul(mul):
    assert isinstance(mul, sp.Mul)

    commutative, other = split_commutative(mul)
    commutative = sp.Mul(*commutative)
    if commutative == 0:
        return S.Zero

    queue = deque(other)
    args = []
    while len(queue) >= 2:
        v1 = queue.popleft()
        v2 = queue.popleft()

        for t in (Transformer_tp_tp, Transformer_sum_tp, Transformer_tp_sum):
            if t.match(v1, v2):
                new_terms = t.transform(v1, v2)
                while new_terms:
                    queue.appendleft(new_terms.pop())
                break
        else:
            debug_warning("fix_misc found unexpected types {} and {} for items {} and {}".format(
                type(v1), type(v2), v1, v2))
            queue.appendleft(v2)
            args.append(v1)

    while queue:
        args.append(queue.popleft())

    return commutative * sp.Mul(*args)


def expand_tp_sum(expr : sp.Sum):
    assert isinstance(expr, sp.Sum)

    function = expr.function
    limits = expr.limits

    if isinstance(function, TP):
        l, r = function.args
        # FIXME: asymmetry in checks
        # also: perhaps come up with a cleaner way to do this?
        if is_y_xy_y(l):
            xy_pow = l.args[1].args[1]
            roots = sp.solve(xy_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if is_y_xy_y(r):
            xy_pow = r.args[1].args[1]
            roots = sp.solve(xy_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if is_x_xy(l):
            xy_pow = l.args[1].args[1]
            roots = sp.solve(xy_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if is_x_xy(r):
            xy_pow = r.args[1].args[1]
            roots = sp.solve(xy_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if is_y_x_xy(l):
            xy_pow = l.args[2].args[1]
            roots = sp.solve(xy_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if is_xy_k_plus_i(l, limits[0]):
            return function.subs(limits[0], 0)
        # FIXME: missing is_xy_k_plus_i(r)
        if matches_xy_k_yx_i(r):
            yx_pow = r.args[1].args[1]
            roots = sp.solve(yx_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
        if matches_yx_i_xy_k(l):
            yx_pow = l.args[0].args[1]
            roots = sp.solve(yx_pow, limits[0])
            assert (len(roots) <= 1)
            if roots:
                return function.subs(limits[0], roots[0])
    else:
        return expr


def expand_tp_impl(expr):
    if isinstance(expr, sp.Add):
        terms = [(expand_tp(arg) if isinstance(arg, sp.Mul) else arg) for arg in expr.args]
        return sp.Add(*terms)
    elif isinstance(expr, sp.Mul):
        return expand_tp_mul(expr)
    elif isinstance(expr, sp.Pow):
        return sp.Pow(expand_tp(expr.args[0]), expr.args[1])
    elif isinstance(expr, sp.Sum):
        return expand_tp_sum(expr)
    elif not (isinstance(expr, TP) or
              isinstance(expr, int) or
              isinstance(expr, sp.Number)):
        raise TypeError("unknown type {} for expression {}".format(type(expr), expr))
    else:
        return expr


def expand_tp(expr, max_iterations=10):
    for _ in range(max_iterations):
        new_expr = expand_tp_impl(expr)
        if expr == new_expr:
            return expr

    raise RuntimeError("No convergence after %d iterations" % max_iterations)


def mysimplify(expr):
    expr = sp.expand(expr, basic=True, mul=True, multinomial=True,
                     power_base=False, power_exp=False, log=False)
    expr = sp.powsimp(expr)
    expr = sp.expand_mul(expr)
    expr = expand_tp(expr)
    expr = sp.expand_mul(expr)
    return expr