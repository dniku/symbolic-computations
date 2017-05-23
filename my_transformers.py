import sympy as sp
from tensor_product import TP
from transformers import Transformer
from my_algebra import x, y, xy, yx, k, c, d, i
import matchers


############################################################
# Transformers derived from my algebra's relations
############################################################

# TODO: derive these automatically from relations

class Transformer_x2(Transformer):
    """ x^2 → y*xy^(k-1) + c*xy^k """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match(v):
        return (
            isinstance(v, sp.Pow) and
            str(v.args[0]) == 'x' and
            str(v.args[1]) == '2'
        )

    @staticmethod
    def transform(v):
        return y * xy ** (k - 1) + c * xy ** k


class Transformer_y2(Transformer):
    """ y^2 → d(xy)^k """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match(v):
        return (
            isinstance(v, sp.Pow) and
            str(v.args[0]) == 'y' and
            str(v.args[1]) == '2'
        )

    @staticmethod
    def transform(v):
        return d * xy ** k


class Transformer_yxk(Transformer):
    """ (yx)^k → (xy)^k """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match(v):
        return (
            isinstance(v, sp.Pow) and
            isinstance(v.args[0], sp.Mul) and
            len(v.args[0].args) == 2 and
            str(v.args[0].args[0]) == 'y' and
            str(v.args[0].args[1]) == 'x' and
            str(v.args[1]) == 'k'
        )

    @staticmethod
    def transform(v):
        return xy ** k


########################################
# Transformers specific to my algebra
########################################

class Transformer_xyi_y(Transformer):
    """ (xy)^i*y → y if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Pow) and
            str(v1.args[0]) == 'x*y' and
            isinstance(v1.args[1], sp.Symbol) and
            str(v2) == 'y'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((y, sp.Eq(v1.args[1], 0)), (0, True))]


class Transformer_y_yxi(Transformer):
    """ y*(yx)^i → y if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            str(v1) == 'y' and
            isinstance(v2, sp.Pow) and
            str(v2.args[0]) == 'y*x' and
            isinstance(v2.args[1], sp.Symbol)
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((y, sp.Eq(v2.args[1], 0)), (0, True))]


class Transformer_yxi_x(Transformer):
    """ (yx)^i*x → x if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Pow) and
            str(v1.args[0]) == 'y*x' and
            isinstance(v1.args[1], sp.Symbol) and
            str(v2) == 'x'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((x, sp.Eq(v1.args[1], 0)), (0, True))]


class Transformer_x_xyi(Transformer):
    """ x*(xy)^i → x if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            str(v1) == 'x' and
            isinstance(v2, sp.Pow) and
            str(v2.args[0]) == 'x*y' and
            isinstance(v2.args[1], sp.Symbol)
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((x, sp.Eq(v2.args[1], 0)), (0, True))]


class Transformer_xyk_yxi(Transformer):
    """ (xy)^{k, k-1}*(yx)^i → (xy)^k if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Pow) and
            isinstance(v2, sp.Pow) and
            str(v1) in {'(x*y)**k', '(x*y)**(k - 1)'} and
            isinstance(v2, sp.Pow) and
            v2.args[0] == 'y*x' and
            isinstance(v2.args[1], sp.Symbol)
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((v1, sp.Eq(v2.args[1], 0)), (0, True))]


class Transformer_k_neq_0_in_piecewise(Transformer):
    """ Piecewise(..., (f, Eq(k, 0)), ...) → Piecewise(...) """

    arg_num = 1
    context = 'none'

    @staticmethod
    def is_k_equals_0(cond):
        return (
            isinstance(cond, sp.Eq) and
            str(cond.args[0]) == 'k' and
            cond.args[1] == 0
        )

    @staticmethod
    def match_transform(v):
        if not isinstance(v, sp.Piecewise):
            return None

        is_k_equals_0 = Transformer_k_neq_0_in_piecewise.is_k_equals_0

        if any(is_k_equals_0(cond) for (f, cond) in v.args):
            new_args = [(f, cond) for (f, cond) in v.args if not is_k_equals_0(cond)]
            return sp.Piecewise(*new_args)
        else:
            return None


########################################
# Sums
########################################


class Transformer_sums(Transformer):
    """ well... """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match_transform(v):
        if not isinstance(v, sp.Sum):
            return None

        function = v.function
        limits = v.limits[0]
        var = limits[0]

        if not isinstance(function, TP):
            return None

        l, r = function.args

        # FIXME: this whole thing
        if matchers.is_y_xy_y(l):
            xy_pow = l.args[1].args[1]
            roots = sp.solve(xy_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.is_y_xy_y(r):
            xy_pow = r.args[1].args[1]
            roots = sp.solve(xy_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.is_x_xy(l):
            xy_pow = l.args[1].args[1]
            roots = sp.solve(xy_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.is_x_xy(r):
            xy_pow = r.args[1].args[1]
            roots = sp.solve(xy_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.is_y_x_xy(l):
            xy_pow = l.args[2].args[1]
            roots = sp.solve(xy_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.is_xy_k_plus_i(l, var):
            return function.subs(var, 0)
        # FIXME: missing is_xy_k_plus_i(r)
        if matchers.matches_xy_k_yx_i(r):
            yx_pow = r.args[1].args[1]
            roots = sp.solve(yx_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])
        if matchers.matches_yx_i_xy_k(l):
            yx_pow = l.args[0].args[1]
            roots = sp.solve(yx_pow, var)
            assert (len(roots) <= 1)
            if roots:
                return function.subs(var, roots[0])

        return None


########################################
# Wrap up and ship
########################################

transformers_relations_general = [
    Transformer_x2,
    Transformer_y2,
    Transformer_yxk,
    Transformer_k_neq_0_in_piecewise,
    # Transformer_sums,
]

transformers_relations_mul = [
    Transformer_xyi_y,
    Transformer_y_yxi,
    Transformer_xyk_yxi,
    Transformer_yxi_x,
    Transformer_x_xyi,
]