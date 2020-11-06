import sympy
from sympy.parsing.sympy_parser import parse_expr


class ClGeneral(object):
    def eval(self, ell, *params):
        assert len(params) == self.n_par
        return self._lambda(ell, *params)

    @property
    def params(self):
        return self._params

    @property
    def n_par(self):
        return len(self._params)

    def _set_default_of_free_symbols(self, **kwargs):
        # Note that
        # - kwargs can contain also keys that are not free symbols
        # - only values of the free symbols are considered
        # - these values are stored in the right order
        self.defaults = [kwargs[symbol] for symbol in self.params]

        @property
        def defaults(self):
            """ Default values of the free parameters
            """
            try:
                assert len(self._defaults) == self.n_param
            except (AttributeError, AssertionError):
                print("Component: unexpected number of or "
                      "uninitialized defaults, returning ones")
                return [1.] * self.n_param
            return self._defaults


class ClAnalytic(ClGeneral):
    def __init__(self, expression, **fixed_params):
        self._fixed_params = fixed_params
        self._expr = parse_expr(expression).subs(fixed_params)
        self._params = sorted([str(s) for s in self._expr.free_symbols])
        self._defaults = []

        # If 'ell' is present, first remove it
        if 'ell' in self._params:
            self._params.pop(self._params.index('ell'))
        # Next add it at the zero-th position
        self._params.insert(0, 'ell')
        # Then create symbols
        symbols = sympy.symbols(self._params)
        # Then remove it again
        self._params.pop(0)

        # Create lambda function
        self._lambda = sympy.lambdify(symbols, self._expr, 'numpy')

    def __repr__(self):
        return repr(self._expr)


class ClPowerLaw(ClAnalytic):
    _REF_ALPHA = -0.5
    _REF_AMP = 1.

    def __init__(self, ell0, amp=None, alpha=None):
        analytic_expr = 'amp * (ell / ell0)**alpha'

        kwargs = {'ell0': ell0, 'alpha': alpha}

        super(ClPowerLaw, self).__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(alpha=self._REF_ALPHA,
                                          amp=self._REF_AMP)
