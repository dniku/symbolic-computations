import sympy as sp


def is_y_xy_y(mul):
    return (
        isinstance(mul, sp.Mul) and
        len(mul.args) == 3 and
        str(mul.args[0]) == 'y' and
        isinstance(mul.args[1], sp.Pow) and
        str(mul.args[1].args[0]) == 'x*y' and
        str(mul.args[2]) == 'y'
    )


def is_x_xy(mul):
    return (
        isinstance(mul, sp.Mul) and
        len(mul.args) == 2 and
        str(mul.args[0]) == 'x' and
        isinstance(mul.args[1], sp.Pow) and
        str(mul.args[1].args[0]) == 'x*y'
    )


def is_y_x_xy(mul):
    return (
        isinstance(mul, sp.Mul) and
        len(mul.args) == 3 and
        str(mul.args[0]) == 'y' and
        str(mul.args[1]) == 'x' and
        isinstance(mul.args[2], sp.Pow) and
        str(mul.args[2].args[0]) == 'x*y'
    )


def is_xy_k_plus_i(pow, counter):
    # matches (xy)^(k+symbol)
    return (
        isinstance(pow, sp.Pow) and
        str(pow.args[0]) == 'x*y' and
        isinstance(pow.args[1], sp.Add) and
        len(pow.args[1].args) == 2 and
        isinstance(pow.args[1].args[0], sp.Symbol) and
        isinstance(pow.args[1].args[1], sp.Symbol) and
        {symbol.name for symbol in pow.args[1].args} == {str(counter), 'k'}
    )


def matches_xy_k_yx_i(mul):
    return (
        isinstance(mul, sp.Mul) and
        len(mul.args) == 2 and
        str(mul.args[0]) == '(x*y)**k' and
        isinstance(mul.args[1], sp.Pow) and
        str(mul.args[1].args[0]) == 'y*x'
    )


def matches_yx_i_xy_k(mul):
    return (
        isinstance(mul, sp.Mul) and
        len(mul.args) == 2 and
        isinstance(mul.args[0], sp.Pow) and
        str(mul.args[0].args[0]) == 'y*x' and
        str(mul.args[1]) == '(x*y)**k'
    )
