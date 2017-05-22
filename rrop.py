# http://docs.sympy.org/dev/guide.html#sympy-s-architecture

from matchers import *
from tensor_product import TP


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
