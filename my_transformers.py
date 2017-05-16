import sympy as sp
from transformers import Transformer
from my_algebra import x, y, xy, yx, k, c, d, i


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
            isinstance(v1.args[0], sp.Mul) and
            len(v1.args[0].args) == 2 and
            str(v1.args[0].args[0]) == 'x' and
            str(v1.args[0].args[1]) == 'y' and
            str(v1.args[1]) == 'i' and
            str(v2) == 'y'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((y, sp.Eq(i, 0)), (0, True))]


class Transformer_y_yxi(Transformer):
    """ y*(yx)^i → y if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            str(v1) == 'y' and
            isinstance(v2, sp.Pow) and
            isinstance(v2.args[0], sp.Mul) and
            len(v2.args[0].args) == 2 and
            str(v2.args[0].args[0]) == 'y' and
            str(v2.args[0].args[1]) == 'x' and
            str(v2.args[1]) == 'i'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((y, sp.Eq(i, 0)), (0, True))]


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
            str(v2) == '(y*x)**i'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((v1, sp.Eq(i, 0)), (0, True))]


class Transformer_yxi_x(Transformer):
    """ (yx)^i*x → x if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            isinstance(v1, sp.Pow) and
            str(v1) == '(y*x)**i' and
            str(v2) == 'x'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((x, sp.Eq(i, 0)), (0, True))]


class Transformer_x_xyi(Transformer):
    """ x*(xy)^i → x if i == 0, 0 otherwise """

    arg_num = 2
    context = 'mul'

    @staticmethod
    def match(v1, v2):
        return (
            str(v1) == 'x' and
            isinstance(v2, sp.Pow) and
            str(v2) == '(x*y)**i'
        )

    @staticmethod
    def transform(v1, v2):
        return [sp.Piecewise((x, sp.Eq(i, 0)), (0, True))]


########################################
# Wrap up and ship
########################################

transformers_relations_general = [
    Transformer_x2,
    Transformer_y2,
    Transformer_yxk,
]

transformers_relations_mul = [
    Transformer_xyi_y,
    Transformer_y_yxi,
    Transformer_xyk_yxi,
    Transformer_yxi_x,
    Transformer_x_xyi,
]