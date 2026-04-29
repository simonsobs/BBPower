from __future__ import annotations

import numpy as np
import sympy
from numpy.typing import ArrayLike
from sympy.parsing.sympy_parser import parse_expr


class ClGeneral:
    """Base class for symbolic angular power spectrum models.

    Subclasses must set ``_params`` (a list of free-parameter names) and
    ``_lambda`` (a callable that evaluates the model) before ``eval`` is
    used.
    """

    def eval(self, ell: ArrayLike, *params: float) -> np.ndarray:
        """Evaluate the power spectrum model.

        Parameters
        ----------
        ell : array_like
            Multipole values at which the model is evaluated.
        *params : float
            Parameter values, one per free parameter, in the order
            returned by ``params``.

        Returns
        -------
        array_like
            Model power spectrum evaluated at the given multipoles.
        """
        assert len(params) == self.n_par
        return self._lambda(ell, *params)

    @property
    def params(self) -> list[str]:
        """list of str : Names of the free parameters."""
        return self._params

    @property
    def n_par(self) -> int:
        """int : Number of free parameters."""
        return len(self._params)

    def _set_default_of_free_symbols(self, **kwargs: float) -> None:
        """Store default values for free parameters.

        Parameters
        ----------
        **kwargs : float
            Keyword arguments whose keys match parameter names in
            ``params``.  Keys that do not correspond to free symbols are
            silently ignored.  Values are stored in the same order as
            ``params``.
        """
        self._defaults = [kwargs[symbol] for symbol in self.params]

    @property
    def defaults(self) -> list[float]:
        """list of float : Default values of the free parameters.

        Returns ones for all parameters if defaults have not been set or
        have an unexpected length.
        """
        try:
            assert len(self._defaults) == self.n_par
        except (AttributeError, AssertionError):
            print(
                "Component: unexpected number of or "
                "uninitialized defaults, returning ones"
            )
            return [1.0] * self.n_par
        return self._defaults


class ClAnalytic(ClGeneral):
    """Analytic power spectrum model built from a string expression.

    The expression is parsed with SymPy.  Any symbol named ``ell`` is
    treated as the multipole variable and becomes the first positional
    argument of the internal lambda.  Remaining free symbols (after
    substituting *fixed_params*) are exposed as model parameters.

    Parameters
    ----------
    expression : str
        A SymPy-parseable mathematical expression (e.g.
        ``'amp * (ell / ell0)**alpha'``).
    **fixed_params : float or None
        Symbol names to substitute with fixed numerical values.  A value
        of ``None`` leaves the symbol free.
    """

    def __init__(self, expression: str, **fixed_params: float | None) -> None:
        self._fixed_params = {k: v for k, v in fixed_params.items() if v is not None}
        self._expr = parse_expr(expression).subs(self._fixed_params)
        self._params = sorted([str(s) for s in self._expr.free_symbols])
        self._defaults = []

        # If 'ell' is present, first remove it
        if "ell" in self._params:
            self._params.pop(self._params.index("ell"))
        # Next add it at the zero-th position
        self._params.insert(0, "ell")
        # Then create symbols
        symbols = sympy.symbols(self._params)
        # Then remove it again
        self._params.pop(0)

        # Create lambda function
        self._lambda = sympy.lambdify(symbols, self._expr, "numpy")

    def __repr__(self) -> str:
        return repr(self._expr)


class ClPowerLaw(ClAnalytic):
    """Power-law Cl model: ``amp * (ell / ell0)**alpha``.

    Parameters
    ----------
    ell0 : float
        Reference (pivot) multipole.
    amp : float or None, optional
        Amplitude.  If ``None`` (default), ``amp`` is left as a free
        parameter.
    alpha : float or None, optional
        Spectral index.  If ``None`` (default), ``alpha`` is left as a
        free parameter.

    Attributes
    ----------
    _REF_ALPHA : float
        Default spectral index used when ``alpha`` is free (-0.5).
    _REF_AMP : float
        Default amplitude used when ``amp`` is free (1.0).
    """

    _REF_ALPHA = -0.5
    _REF_AMP = 1.0

    def __init__(
        self, ell0: float, amp: float | None = None, alpha: float | None = None
    ) -> None:
        analytic_expr = "amp * (ell / ell0)**alpha"

        kwargs = {"ell0": ell0, "alpha": alpha}

        super().__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(alpha=self._REF_ALPHA, amp=self._REF_AMP)
