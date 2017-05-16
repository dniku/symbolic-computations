import abc

import sympy as sp

from tensor_product import TP


class Transformer(object):
    __metaclass__ = abc.ABCMeta

    # `self` is in the parameter list to make PyCharm happy

    @property
    @abc.abstractstaticmethod
    def arg_num(self) -> int:
        pass

    @property
    @abc.abstractstaticmethod
    def context(self) -> str:
        """ One of: 'none', 'mul' (maybe 'add' in the future) """
        pass

    @abc.abstractstaticmethod
    def match(self, *args) -> bool:
        pass

    @abc.abstractstaticmethod
    def transform(self, *args):
        pass


########################################
# Basic arithmetic
########################################

class Transformer_x_y_x_y(Transformer):
    """ x*y*x*y → (x*y)^2 """

    arg_num = 4
    context = 'mul'

    @staticmethod
    def match(v1, v2, v3, v4):
        return (
            isinstance(v1, sp.Symbol) and
            isinstance(v2, sp.Symbol) and
            v1 == v3 and
            v2 == v4
        )

    @staticmethod
    def transform(v1, v2, v3, v4):
        return [(v1 * v2)**2]


class Transformer_xy_x_y(Transformer):
    """ (xy)^i * x * y → (xy)^(i+1) """

    arg_num = 3
    context = 'mul'

    @staticmethod
    def match(v1, v2, v3):
        return (
            isinstance(v1, sp.Pow) and
            isinstance(v1.args[0], sp.Mul) and
            len(v1.args[0].args) == 2 and
            v1.args[0].args[0] == v2 and
            v1.args[0].args[1] == v3
        )

    @staticmethod
    def transform(v1, v2, v3):
        return [(v2 * v3) ** (v1.args[1] + 1)]


class Transformer_y_xy_x(Transformer):
    """ y * (xy)^i * x → (yx)^(i+1) """

    arg_num = 3
    context = 'mul'

    @staticmethod
    def match(v1, v2, v3):
        return Transformer_xy_x_y.match(v2, v3, v1)

    @staticmethod
    def transform(v1, v2, v3):
        return [(v1 * v3) ** (v2.args[1] + 1)]


class Transformer_x_y_xy(Transformer):
    """ x * y * (xy)^i → (xy)^(i+1) """

    arg_num = 3
    context = 'mul'

    @staticmethod
    def match(v1, v2, v3):
        return Transformer_xy_x_y.match(v3, v1, v2)

    @staticmethod
    def transform(v1, v2, v3):
        return [(v1 * v2) ** (v3.args[1] + 1)]


class Transformer_xy_x(Transformer):
    """ (xy)^i * x → x * (yx)^i """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Pow) and
            isinstance(v1.args[0], sp.Mul) and
            len(v1.args[0].args) == 2 and
            v1.args[0].args[0] == v2
        )

    @staticmethod
    def transform(v1, v2):
        return [
            v1.args[0].args[0],
            (v1.args[0].args[1] * v2) ** v1.args[1]
        ]


########################################
# TPs (tensor products)
########################################

class Transformer_tp2(Transformer):
    """ TP(a, b)^2 → TP(a^2, b^2) """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match(v):
        return (
            isinstance(v, sp.Pow) and
            isinstance(v.args[0], TP) and
            v.args[1] == 2
        )

    @staticmethod
    def transform(v: sp.Pow):
        tp = v.args[0]
        l = tp.args[0] * tp.args[0]
        r = tp.args[1] * tp.args[1]
        return TP(l, r)


class Transformer_tp_tp(Transformer):
    """ TP(a, b) * TP(c, d) → TP(a * c, d * b) """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, TP) and
            isinstance(v2, TP)
        )

    @staticmethod
    def transform(v1 : TP, v2 : TP):
        l = v1.args[0] * v2.args[0]
        r = v2.args[1] * v1.args[1]
        return [TP(l, r)]


class Transformer_tp_sum(Transformer):
    """ TP(a, b) * Sum(f, limits) → Sum(TP(a, b) * f, limits) """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, TP) and
            isinstance(v2, sp.Sum)
        )

    @staticmethod
    def transform(v1 : TP, v2 : sp.Sum):
        return [sp.Sum(v1 * v2.function, v2.limits[0])]


class Transformer_sum_tp(Transformer):
    """ Sum(f, limits) * TP(a, b) → Sum(f * TP(a, b), limits) """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Sum) and
            isinstance(v2, TP)
        )

    @staticmethod
    def transform(v1 : sp.Sum, v2 : TP):
        return [sp.Sum(v1.function * v2, v1.limits[0])]

########################################
# Piecewise functions
########################################

class Transformer_piecewise_one_eq(Transformer):
    """ Piecewise(..., (f, Eq(x, val)), ...) → Piecewise(..., (f.subs(x, val), Eq(x, val)), ...) """

    arg_num = 1
    context = 'none'

    @staticmethod
    def match(v):
        return (
            isinstance(v, sp.Piecewise) and
            any(isinstance(arg[1], sp.Eq) for arg in v.args)
        )

    @staticmethod
    def transform(v):
        def transform_pair(f, cond):
            if isinstance(cond, sp.Eq):
                x, val = cond.args
                return (f.subs(x, val), sp.Eq(x, val))
            else:
                return (f, cond)
        args = [transform_pair(*arg) for arg in v.args]
        return sp.Piecewise(*args)


########################################
# Piecewise functions within sequences
########################################

class Transformer_seq_piecewise_out_of_bounds(Transformer):
    """ SeqFormula(Piecewise(..., (f, Eq(x, v0)), ...), (x, v1, v2)) where v0 < v1 or v2 < v0
      → SeqFormula(Piecewise(...), (x, v1, v2)) """

    arg_num = 1
    context = 'none'

    @staticmethod
    def good_cond(cond, limits):
        if not isinstance(cond, sp.Eq):
            return True
        x, v0 = cond.args
        if x != limits[0]:
            return True
        _, v1, v2 = limits
        try:
            return not (v0 < v1 or v2 < v0)
        except TypeError:
            return True

    @staticmethod
    def match(v):
        good_cond = Transformer_seq_piecewise_out_of_bounds.good_cond
        return (
            isinstance(v, sp.SeqFormula) and
            isinstance(v.formula, sp.Piecewise) and
            any(not good_cond(arg[1], v.args[1]) for arg in v.formula.args)
        )

    @staticmethod
    def transform(v):
        good_cond = Transformer_seq_piecewise_out_of_bounds.good_cond
        args = [(f, cond) for (f, cond) in v.formula.args if good_cond(cond, v.args[1])]
        return sp.SeqFormula(sp.Piecewise(*args), v.args[1])


########################################
# Wrap up and ship
########################################

transformers_general = [
    Transformer_piecewise_one_eq,
    Transformer_seq_piecewise_out_of_bounds,
    Transformer_tp2,
]

transformers_mul = [
    Transformer_x_y_x_y,  # x*y*x*y → (x*y)^2
    Transformer_xy_x_y,  # (xy)^i * x * y → (xy)^(i+1)
    Transformer_y_xy_x,  # y * (xy)^i * x → (yx)^(i+1)
    Transformer_x_y_xy,  # x * y * (xy)^i → (xy)^(i+1)
    Transformer_xy_x,  # (xy)^i * x → x * (yx)^i
    Transformer_tp_tp,
    Transformer_tp_sum,
    Transformer_sum_tp,
]