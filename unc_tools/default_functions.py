"""Symbolic helper utilities with uncertainty-aware coefficients."""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Sequence

import numpy as np
import re
import sympy as sym
import uncertainties as unc

from .exceptions import ExpressionError

__all__ = ["FunctionBase1D", "Hyper", "Poly"]

Uncertain = unc.core.AffineScalarFunc | unc.core.Variable
Numeric = float | int | np.number
Coefficient = sym.Expr | Numeric | Uncertain
CoefficientsInput = Sequence[Coefficient] | np.ndarray
Solution = sym.Expr | Uncertain | tuple[Uncertain, Uncertain]
LambdaReturn = float | complex | np.ndarray | Uncertain
InputValue = Numeric | np.ndarray | Uncertain


class FunctionBase1D:
    """Represent a one-dimensional symbolic expression with optional uncertainty data."""

    def __init__(self, expr_str: str = "") -> None:
        """Initialize the expression and extract coefficient symbols.

        Args:
            expr_str: String representation of the sympy expression to manage.

        Raises:
            sym.SympifyError: If the expression string cannot be parsed.
        """
        self.expr_str = expr_str
        self.expr_unc: str = ""
        self.expr_sym = sym.parse_expr(expr_str)
        self.args = self.ordered_symbols(expr_str)

        self._added_coefs = False
        self._unc_coefs = False
        self._complex = False
        self._show_complex = False

        return None

    @staticmethod
    def ordered_symbols(expr_str: str) -> list[sym.Symbol]:
        """Return sympy symbols in the order they appear in the expression string.

        Args:
            expr_str: Expression string to scan.

        Returns:
            A list of symbols ordered by their first appearance in the expression.
        """
        expr = sym.sympify(expr_str)

        symbols = expr.free_symbols

        name_to_symbol = {str(s): s for s in symbols}

        ordered = []
        used = set("x")

        i = 0
        n = len(expr_str)

        while i < n:
            for name, symb in name_to_symbol.items():
                if expr_str.startswith(name, i):
                    if name not in used:
                        ordered.append(symb)
                        used.add(name)
                    i += len(name)
                    break
            else:
                i += 1

        return ordered

    @cached_property
    def coefs(self) -> list[Coefficient]:
        """Return coefficient placeholders or bound values.

        Before coefficients are bound, this list contains sympy symbols in a stable
        order. After calling `add_coefs`, it contains the supplied coefficient values.

        Returns:
            A list of coefficient symbols or values.
        """
        return self.args

    @cached_property
    def sols(self) -> list[Solution]:
        """Solve the expression set equal to zero.

        Returns analytical solutions, optionally excluding complex values unless
        complex display is enabled.

        Returns:
            A list of solutions for the equation ``expr_sym = 0``.
        """
        # try:
        #     solution = sym.solvers.solve(self.expr_sym - 0, sym.Symbol("x"))
        #
        #     # if not solution:
        #     #    raise TypeError("Cant find analytical solution for given expression")
        #
        #     if self._show_complex:
        #         return solution
        #     else:
        #         solution = [x for x in solution if not x.is_imaginary]
        #         return solution
        # except:
        #     return []
        #
        solution = sym.solvers.solve(self.expr_sym - 0, sym.Symbol("x"))

        # if not solution:
        #    raise TypeError("Cant find analytical solution for given expression")

        if self._show_complex:
            return solution
        else:
            solution = [x for x in solution if not x.is_imaginary]
            return solution

    def find_sols(
        self, y: int | float | sym.Expr | str | Uncertain = 0
    ) -> list[Solution] | Solution:
        """Solve the expression for a specified target value.

        Args:
            y: Target value for ``expr_sym = y``. Strings are parsed as sympy expressions.

        Returns:
            A list of solutions or a single solution when only one exists.
        """
        if isinstance(y, (unc.core.Variable, unc.core.AffineScalarFunc)):
            new_expr = FunctionBase1D(self.expr_str + " - NEW_1")

            if isinstance(self.coefs, np.ndarray):
                self.coefs = self.coefs.tolist()
            # print(type(self.coefs))
            # print(y)
            list_coefs = self.coefs + [y]
            # print(new_expr.coefs)
            # print(list_coefs)
            if self._added_coefs:
                new_expr.add_coefs(list_coefs)

        else:
            if isinstance(y, str):
                y = sym.parse_expr(y)
            new_expr = FunctionBase1D(self.expr_str)
            new_expr.expr_sym -= y
            if self._added_coefs:
                new_expr.add_coefs(self.coefs)

        if len(new_expr.sols) > 1 and hasattr(new_expr.sols, "__iter__"):
            return new_expr.sols
        elif len(new_expr.sols) == 0:
            return []
        else:
            return new_expr.sols[0]

    @cached_property
    def lambda_fun(self) -> Callable[..., LambdaReturn]:
        """Generate a numpy-evaluable callable for the expression.

        Returns:
            A callable accepting ``x`` followed by coefficient values.
        """
        lambda_args = [sym.Symbol("x")] + self.args
        return sym.lambdify(lambda_args, self.expr_sym, "numpy")

    def show_complex(self, show: bool = False) -> "FunctionBase1D":
        """Toggle the display of complex solutions.

        Args:
            show: Whether to include complex-valued solutions.

        Returns:
            The current instance with updated display preference.
        """
        self._show_complex = show

        return self

    @staticmethod
    def _calculate_uncertainty_analyticaly(
        expr: sym.Expr,
        parms: Sequence[sym.Symbol] | None = None,
    ) -> sym.Expr:
        """Compute propagated uncertainty for a symbolic expression.

        Args:
            expr: Expression whose uncertainty is estimated.
            parms: Parameters treated as uncertain; defaults to free symbols.

        Returns:
            A symbolic expression representing the combined uncertainty.
        """
        if parms is None:
            parms = expr.free_symbols

        unc_list: list[sym.Expr] = []
        for parm in parms:
            unc_list.append(
                (sym.diff(expr, parm) * sym.Symbol(f"Delta_{str(parm)}")) ** 2
            )

        return sym.sqrt(np.sum(unc_list))

    def add_coefs(self, coefs: CoefficientsInput) -> "FunctionBase1D":
        """Apply coefficients to the expression and rebuild cached artifacts.

        Args:
            coefs: Coefficient values matching the detected symbols.

        Returns:
            The current instance with coefficients applied.

        Raises:
            ExpressionError: If the number of coefficients differs from expectations.
        """
        if len(coefs) != len(self.coefs):
            raise ExpressionError(
                "Number of arguments does not match the number of coefficients."
            )

        self.coefs = coefs

        if any(
            [
                isinstance(x, (unc.core.Variable, unc.core.AffineScalarFunc))
                for x in coefs
            ]
        ):
            self._unc_coefs = True

        coefs_dict = {}
        for i in range(len(coefs)):
            coefs_dict[self.args[i]] = coefs[i]

        if self._unc_coefs:
            # Separating nominal values and standard deviations.
            nominal_coefs_dict = {}
            std_coefs_dict = {}
            for key in coefs_dict:
                delta = sym.Symbol(f"Delta_{str(key)}")
                if isinstance(
                    coefs_dict[key], (unc.core.Variable, unc.core.AffineScalarFunc)
                ):
                    nominal_coefs_dict[key] = coefs_dict[key].nominal_value
                    std_coefs_dict[delta] = coefs_dict[key].std_dev
                else:
                    nominal_coefs_dict[key] = coefs_dict[key]
                    std_coefs_dict[delta] = 0

            # making solutions
            sols_list = []
            for sol in self.sols:
                # print(sol)
                nominal = sol.subs(nominal_coefs_dict)
                # print(nominal)
                if nominal.is_complex and not nominal.is_real:
                    self._complex = True

                    if self._show_complex:
                        std = self._calculate_uncertainty_analyticaly(sol).subs(
                            {**nominal_coefs_dict, **std_coefs_dict}
                        )
                        real_part = unc.ufloat(sym.re(nominal), abs(sym.re(std)))
                        im_part = unc.ufloat(sym.im(nominal), abs(sym.im(std)))
                        sols_list.append((real_part, im_part))

                else:
                    std = self._calculate_uncertainty_analyticaly(sol).subs(
                        {**nominal_coefs_dict, **std_coefs_dict}
                    )
                    sols_list.append(unc.ufloat(nominal, std))

            self.sols = sols_list

            # making lambda function
            def wrap_lambdify(
                func: Callable[..., LambdaReturn], *coefs: Coefficient
            ) -> Callable[[InputValue], LambdaReturn]:
                def new_lambda(x: InputValue) -> LambdaReturn:
                    return func(x, *coefs)

                return new_lambda

            self.lambda_fun = wrap_lambdify(
                sym.lambdify([sym.Symbol("x")] + self.args, self.expr_sym, "numpy"),
                *coefs,
            )

            # making str expr
            self.expr_unc = self.expr_str
            for key in coefs_dict:
                if isinstance(coefs_dict[key], unc.core.Variable):
                    replacement = f"unc.ufloat({coefs_dict[key].nominal_value},{coefs_dict[key].std_dev})"
                else:
                    replacement = str(coefs_dict[key])

                self.expr_unc = str(self.expr_unc).replace(str(key), replacement)

            self._added_coefs = True

        else:
            self.expr_sym = self.expr_sym.subs(coefs_dict)
            self.lambda_fun = sym.lambdify(sym.Symbol("x"), self.expr_sym, "numpy")
            self._added_coefs = True

            sols = self.sols
            self.sols = [x.subs(coefs_dict) for x in sols]
            if any([x.is_complex for x in self.sols]):
                self._complex = True

        return self

    def to_latex_expr(self) -> str:
        """Convert the expression (with coefficients) to a LaTeX string.

        Returns:
            LaTeX-formatted expression string.
        """
        latex = sym.latex(self.expr_sym)

        if self._added_coefs:
            for i in range(len(self.coefs)):
                if isinstance(self.coefs[i], unc.core.Variable):
                    replacement = f"({self.coefs[i]})"
                else:
                    replacement = f"{self.coefs[i]}"

                latex = latex.replace(sym.latex(self.args[i]), replacement)

            latex = latex.replace("+/-", r" \pm ")
        latex = re.sub(
            r"e([+-])(\d+)",
            lambda m: rf"\cdot 10^{{{m.group(1)}{str(int(m.group(2)))}}}",
            latex,
        )
        return rf"{latex}"

    def _calculate_sols(self, show_unc: bool | None = None) -> list[str]:
        """Prepare LaTeX solution strings with optional uncertainty notation.

        Args:
            show_unc: Whether to include propagated uncertainties.

        Returns:
            A list of LaTeX-formatted solution strings.
        """
        latex_sols = []

        if self._added_coefs:
            if self._show_complex:
                if self.sols:
                    del self.__dict__["sols"]
                self.add_coefs(self.coefs)
                latex_sols = [
                    str(x) if not isinstance(x, tuple) else f"({x[0]}) + ({x[1]}) i"
                    for x in self.sols
                ]

            else:
                latex_sols = self.sols

            latex_sols = [f"{x}".replace("+/-", " \\pm ") for x in latex_sols]

        else:
            if show_unc:
                if self.sols:
                    del self.__dict__["sols"]

                latex_sols = [
                    f"{sym.latex(x)} + {sym.latex(self._calculate_uncertainty_analyticaly(x))}"
                    for x in self.sols
                ]

            else:
                if self.sols:
                    del self.__dict__["sols"]
                latex_sols = [sym.latex(x) for x in self.sols]

        return latex_sols

    def to_latex_sols(
        self, y: Coefficient | str | None = None, show_unc: bool = False
    ) -> str:
        """Format solutions as a LaTeX array with an optional target offset.

        Args:
            y: Optional target value to subtract prior to solving.
            show_unc: Whether to include uncertainties in the output.

        Returns:
            LaTeX-formatted solution array.
        """
        show_unc = show_unc if not self._unc_coefs else self._unc_coefs

        if not y:
            latex_sols = self._calculate_sols(show_unc=show_unc)
        elif isinstance(y, str):
            new_expr = FunctionBase1D(str(self.expr_sym) + " - " + y)
            latex_sols = new_expr._calculate_sols(show_unc=show_unc)
        elif isinstance(y, unc.core.Variable) or isinstance(
            y, unc.core.AffineScalarFunc
        ):
            new_expr = FunctionBase1D(self.expr_str + " - A100")
            new_expr.add_coefs(self.coefs + [y])
            latex_sols = new_expr._calculate_sols(show_unc=show_unc)
        else:
            new_expr = FunctionBase1D(self.expr_str + "-" + str(y))
            if self._added_coefs:
                new_expr.add_coefs(self.coefs)

            latex_sols = new_expr._calculate_sols(show_unc=show_unc)

        for i, sol in enumerate(latex_sols):
            latex_sols[i] = f"x_{i + 1} = {sol} \\\\"

        lines = f"\\begin{{array}}{{l}}\n{'\n'.join(latex_sols)}\n\\end{{array}}"
        lines = re.sub(
            r"e([+-])(\d+)",
            lambda m: rf"\cdot 10^{{{m.group(1)}{str(int(m.group(2)))}}}",
            lines,
        )
        return lines

    def deriv(self, level: int = 1) -> "FunctionBase1D":
        """Compute a derivative of the expression.

        Args:
            level: Order of the derivative to compute.

        Returns:
            New instance representing the derivative.
        """
        expr = self.expr_sym

        deriv = sym.diff(expr, sym.Symbol("x"), level)

        new_expr_str = str(deriv)

        new_func = FunctionBase1D(new_expr_str)

        if self._added_coefs:
            new_func.add_coefs(self.coefs)

        return new_func


class Poly(FunctionBase1D):
    """Construct a polynomial expression of the specified degree."""

    def __init__(self, degree: int = 1) -> None:
        """Create a polynomial expression of the specified degree.

        Args:
            degree: Polynomial degree.
        """
        self.degree = degree

        expr_str = ""
        for i in range(degree):
            expr_str += f"p_{i}*x**{degree - i} + "
        expr_str += f"p_{degree}"

        super().__init__(expr_str)
        # self.expr_str = expr_str
        return None

    @cached_property
    def sols(self) -> list[Solution]:
        """Return analytical solutions for the polynomial when solvable.

        Returns:
            A list of solutions for the polynomial equation.

        Raises:
            ExpressionError: If the polynomial degree exceeds four.
        """
        if self.degree > 4:
            raise ExpressionError(
                "Cannot find analytical solutions for polynomials with degree > 4."
            )
        else:
            return super().sols


class Hyper(FunctionBase1D):
    """Construct a hyperbolic expression variant."""

    def __init__(self, style: int = 0) -> None:
        """Initialize a hyperbolic expression variant.

        Args:
            style: Expression selector; ``0`` uses ``p_0*x / (x + p_1)``, otherwise
                uses ``p_0 / (x + p_1) + p_2``.
        """
        if style == 0:
            expr_str = "p_0*x / (x + p_1)"
        else:
            expr_str = "p_0 / (x + p_1) + p_2"

        super().__init__(expr_str)
        return None
