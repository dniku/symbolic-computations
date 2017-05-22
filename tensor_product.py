import sympy as sp
from sympy import Expr, S
from sympy.printing.pretty.stringpict import prettyForm


class TP(Expr):
    __slots__ = ['is_commutative']  # all base classes have __slots__, hence need this
    # _op_priority = 100.0  # integers have 10.0

    def __new__(cls, l, r):
        l = sp.sympify(l)
        r = sp.sympify(r)
        obj = Expr.__new__(cls, l, r)
        obj.is_commutative = False
        return obj

    def _pretty(self, *args, **kwargs):
        """ Used in terminal. """
        # FIXME: powers are placed incorrectly
        printer, = args
        lstr, rstr = [printer.doprint(arg) for arg in self.args]
        return prettyForm('(%s⊗%s)' % (lstr, rstr))

    def _latex(self, *args, **kwargs):
        """ Used in Notebook. """
        printer, = args
        lstr, rstr = [printer.doprint(arg) for arg in self.args]
        return '\\left( %s \\otimes %s \\right)' % (lstr, rstr)
