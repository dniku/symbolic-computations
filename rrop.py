# http://docs.sympy.org/dev/guide.html#sympy-s-architecture
# https://github.com/sympy/sympy/search?p=2&q=Xor&utf8=%E2%9C%93

from collections import deque
import sympy as sp
from sympy.core.expr import Expr
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.numbers import Zero

DEBUG = False
WARNINGS = True

k = sp.symbols('k')
x, y = sp.symbols('x, y', commutative=False)
c, d = sp.symbols('c, d')
xy = x * y
yx = y * x

relations = {
    'x**2': y * xy ** (k - 1) + c * xy ** k,
    'y**2': d * xy ** k,
    '(y*x)**k': xy ** k
}

zeros_mul = {
    '(x*y)**(k - 1)*y',
    'x*(x*y)**(k - 1)',
    'y*(y*x)**(k - 1)',
    'x**2*y',
    'x*(x*y)**k',
    '(x*y)**k*y',
    'y*(x*y)**k',
    '(y*x)**(k - 1)*x',
    # TODO: get rid at least of these
    'y*(x*y)**(2*k - 2)',
    'y*(x*y)**(k - 2)*y*(x*y)**(k - 1)'
}

zeros_pow = {
    '(x*y)**(2*k - 1)',
}


def matches_xy_x_y(v1, v2, v3):
    # (xy)^i * xy → (xy)^(i+1)
    return (
        isinstance(v1, sp.Pow) and
        isinstance(v1.args[0], sp.Mul) and
        len(v1.args[0].args) == 2 and
        str(v1.args[0].args[0]) == str(v2) and
        str(v1.args[0].args[1]) == str(v3)
    )


def matches_xy_x(v1, v2):
    return (
        isinstance(v1, sp.Pow) and
        isinstance(v1.args[0], sp.Mul) and
        len(v1.args[0].args) == 2 and
        str(v1.args[0].args[0]) == str(v2)
    )


def canonical_mul(expr):
    # fix triples
    queue = deque(expr.args)
    args = []
    while len(queue) >= 3:
        v1 = queue.popleft()
        v2 = queue.popleft()
        v3 = queue.popleft()

        if matches_xy_x_y(v1, v2, v3):
            # (xy)^i * xy → (xy)^(i+1)
            queue.appendleft((v2 * v3) ** (v1.args[1] + 1))
        elif matches_xy_x_y(v2, v3, v1):
            # y * (xy)^i * x → (yx)^(i + 1)
            queue.appendleft((v1 * v3) ** (v2.args[1] + 1))
        elif matches_xy_x_y(v3, v1, v2):
            # xy * (xy)^i → (xy)^(i+1)
            queue.appendleft((v1 * v2) ** (v3.args[1] + 1))
        else:
            queue.appendleft(v3)
            queue.appendleft(v2)
            args.append(v1)
    while queue:
        args.append(queue.popleft())

    # fix duplets
    queue = deque(args)
    args = []
    while len(queue) >= 2:
        v1 = queue.popleft()
        v2 = queue.popleft()

        if matches_xy_x(v1, v2):
            # (xy)^i * x → x * (yx)^i
            yy = v1.args[0].args[1]  # 'matches' does not mean 'equals', so 'yy' could be 'x'
            queue.appendleft((yy * v2) ** v1.args[1])
            queue.appendleft(v2)
        else:
            queue.appendleft(v2)
            args.append(v1)
    while queue:
        args.append(queue.popleft())

    def catch_zeros(expr):
        s = str(expr)
        for z in zeros_mul:
            if z in s:
                return 0
        return expr

    for i, arg in enumerate(args):
        s = str(arg)
        for r in relations.keys():
            if s == r:
                # WARNING: this may miss zeros produced by x**2 if canonical_mul is not restarted
                return catch_zeros(sp.Mul(*(args[:i] + [relations[r]] + args[i + 1:])))

    return catch_zeros(sp.Mul(*args))


def canonical_expr(expr):
    if DEBUG:
        return expr
    expr = sp.powsimp(expr)
    if isinstance(expr, sp.Number):
        return expr % 2
    elif isinstance(expr, sp.Symbol):
        return expr
    elif isinstance(expr, sp.Add):
        return expr
    elif isinstance(expr, sp.Mul):
        return canonical_mul(expr)
    elif isinstance(expr, sp.Pow):
        s = str(expr)
        for z in zeros_pow:
            if z in s:
                return 0
        for r in relations.keys():
            if s == r:
                return relations[r]
        return expr
    else:
        print("%s is unmatched type %s" % (expr, type(expr)))
        return expr


def split_coefficients(expr):
    if not isinstance(expr, sp.Mul):
        return 1, expr
    variables = []
    coefficients = []
    for arg in expr.args:
        if isinstance(arg, sp.Number) or isinstance(arg, int):
            coefficients.append(arg % 2)
        elif arg == c or arg == d:
            # FIXME: doesn't handle things like c**2
            coefficients.append(arg)
        else:
            variables.append(arg)
    return sp.Mul(*coefficients), sp.Mul(*variables)


def tp_times_tp(tp1, tp2):
    l = tp1.args[0] * tp2.args[0]
    r = tp2.args[1] * tp1.args[1]
    return TP(l, r)


def tp_times_sum(tp, s):
    return Sum(tp * s.function, s.limits[0])


def sum_times_tp(s, tp):
    return Sum(s.function * tp, s.limits[0])


class TP(Expr):
    __slots__ = ['is_commutative']  # all base classes have __slots__, hence need this
    _op_priority = 100.0  # integers have 10.0

    def __new__(cls, l, r):
        l = canonical_expr(sp.sympify(l))
        r = canonical_expr(sp.sympify(r))
        if l == Zero or r == Zero or l == 0 or r == 0:
            return 0  # Zero prints as <class 'sympy.core.numbers.Zero'>

        if isinstance(l, sp.Add):
            return sum(TP(arg, r) for arg in l.args)
        if isinstance(r, sp.Add):
            return sum(TP(l, arg) for arg in r.args)

        # to make PyCharm happy
        assert isinstance(l, Expr)
        assert isinstance(r, Expr)

        lc, lv = split_coefficients(l)
        rc, rv = split_coefficients(r)

        obj = Expr.__new__(cls, lv, rv)
        obj.is_commutative = False
        return (lc * rc) * obj

    def __neg__(self):
        return self

    def __mul__(self, other):
        other = sp.sympify(other)
        if isinstance(other, TP):
            return tp_times_tp(self, other)
        elif isinstance(other, sp.Sum):
            return tp_times_sum(self, other)
        elif other.is_number:
            if other % 2 == 0:
                return 0
            else:
                return self
        else:
            return sp.Mul(self, other)

    def __rmul__(self, other):
        other = sp.sympify(other)
        if isinstance(other, TP):
            return tp_times_tp(other, self)
        elif isinstance(other, sp.Sum):
            return sum_times_tp(other, self)
        elif other.is_number:
            if other % 2 == 0:
                return 0
            else:
                return self
        else:
            return sp.Mul(other, self)

    def __pow__(self, power):
        if power == 2:
            return tp_times_tp(self, self)
        return sp.Pow(self, power)

    def _pretty(self, *args, **kwargs):
        """ Used in terminal. """
        # FIXME: powers are all over the place (incorrect placement)
        printer, = args
        lstr, rstr = [printer.doprint(arg) for arg in self.args]
        return prettyForm('(%s⊗%s)' % (lstr, rstr))

    def _latex(self, *args, **kwargs):
        """ Used in Notebook. """
        printer, = args
        lstr, rstr = [printer.doprint(arg) for arg in self.args]
        return '\\left( %s \\otimes %s \\right)' % (lstr, rstr)


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


class Sum(sp.Sum):
    def __new__(cls, function, limits):
        function = sp.sympify(function)
        if function == 0:
            return 0
        if isinstance(function, TP):
            l, r = function.args
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
            # FIXME: missing is_x_xy(r)
            if is_xy_k_plus_i(l, limits[0]):
                return function.subs(limits[0], 0)
        elif isinstance(function, sp.Add):
            return sum(Sum(arg, limits) for arg in function.args)
        elif isinstance(function, sp.Mul):
            fc, fv = split_coefficients(function)
            return fc * Sum(fv, limits)

        return sp.Sum.__new__(cls, function, limits)


def fix_sums(expr):
    if DEBUG:
        return expr
    if not expr.args:
        return expr
    args = [fix_sums(arg) for arg in expr.args]
    if isinstance(expr, sp.Sum):
        return Sum(*args)
    else:
        try:
            return expr.func(*args)
        except TypeError as e:
            print('failure with type %s' % type(expr))
            raise e


def debug_warning(s):
    if WARNINGS:
        print('WARNING:', s)


def fix_misc_mul(mul):
    assert isinstance(mul, sp.Mul)

    coefficients = []
    args = []
    for arg in mul.args:
        if isinstance(arg, int) or isinstance(arg, sp.Number) or arg.is_commutative:
            if isinstance(arg, int) or isinstance(arg, sp.Number):
                arg_fixed = arg % 2
                if arg != arg_fixed:
                    # The following warning throws too often:
                    # debug_warning("fix_misc fixed number {} to {}".format(arg, arg_fixed))
                    if arg_fixed == 0:
                        return 0
                arg = arg_fixed
            coefficients.append(arg)
        elif isinstance(arg, sp.Sum):
            args.append(Sum(*arg.args))
        else:
            args.append(arg)

    # fix duplets
    queue = deque(args)
    args = []
    while len(queue) >= 2:
        v1 = queue.popleft()
        v2 = queue.popleft()

        if isinstance(v1, TP) and isinstance(v2, TP):
            debug_warning("fix_misc found TP*TP in a Mul: {} and {}".format(v1, v2))
            queue.appendleft(tp_times_tp(v1, v2))
        elif isinstance(v1, TP) and isinstance(v2, Sum):
            debug_warning("fix_misc found TP*Sum in a Mul: {} and {}".format(v1, v2))
            queue.appendleft(tp_times_sum(v1, v2))
        elif isinstance(v1, Sum) and isinstance(v2, TP):
            debug_warning("fix_misc found Sum*TP in a Mul: {} and {}".format(v1, v2))
            queue.appendleft(sum_times_tp(v1, v2))
        else:
            debug_warning("fix_misc found unexpected types {} and {} for items {} and {}".format(
                type(v1), type(v2), v1, v2))
            queue.appendleft(v2)
            args.append(v1)
    while queue:
        args.append(queue.popleft())

    return sp.Mul(*coefficients) * sp.Mul(*args)


def fix_misc(expr):
    if DEBUG:
        return expr

    if isinstance(expr, sp.Add):
        args = []
        for arg in expr.args:
            if isinstance(arg, sp.Mul):
                new_arg = fix_misc_mul(arg)
            else:
                new_arg = arg
            args.append(new_arg)
        return sp.Add(*args)
    elif not (isinstance(expr, sp.Mul) or
                  isinstance(expr, TP) or
                  isinstance(expr, Sum) or
                  isinstance(expr, int) or
                  isinstance(expr, sp.Number)):
        raise TypeError("fix_misc received an unknown type {} for expression {}".format(type(expr), expr))
    else:
        return expr


def mysimplify(expr):
    expr = sp.expand(expr, basic=True, mul=True, multinomial=True,
                     power_base=False, power_exp=False, log=False)
    expr = sp.powsimp(expr)
    expr = fix_sums(expr)
    expr = sp.expand_mul(expr)
    expr = fix_misc(expr)
    return expr
