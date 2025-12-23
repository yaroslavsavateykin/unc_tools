from ast import Call, Dict, Raise, expr
from functools import cached_property
from math import exp
from os import replace
from typing import Callable, List, Union, Any, Optional
from matplotlib.pyplot import show
import sympy as sym
from sympy.matrices.expressions.matmul import new
import uncertainties as unc
import numpy as np
import re


class FunctionBase1D:
    def __init__(self, expr_str: str = "") -> None:
        """Initialize a one-dimensional symbolic expression handler.

        Parses the provided expression string, extracts coefficients (excluding ``x``),
        and initializes flags controlling uncertainty handling and complex solution
        visibility without modifying the expression logic.

        Args:
            expr_str (str): String representation of the sympy expression to manage.

        Returns:
            None

        Raises:
            sym.SympifyError: If the expression string cannot be parsed.

        Side Effects:
            Sets internal state for later calculations.

        Examples:
            >>> FunctionBase1D("a*x + b")
        """
        self.expr_str = expr_str
        self.expr_unc: str = ""
        self.expr_sym = sym.parse_expr(expr_str)
        self.args = sorted(
            [x for x in list(self.expr_sym.free_symbols) if not str(x) == "x"],
            key=lambda s: str(s),
        )

        self._added_coefs = False
        self._unc_coefs = False
        self._complex = False
        self._show_complex = False

        # self.expr: Union[sym.core.add.Add, sym.core.expr.Expr]
        # self.coefs: List[sym.core.symbol.Symbol]
        # self.sols = List[sym.core.mul.Mul]
        # self.lambda_fun: Callable

    @cached_property
    def coefs(self) -> List[sym.core.symbol.Symbol]:
        """Return coefficient symbols sorted for consistency.

        Returns:
            list[sym.core.symbol.Symbol]: Sorted coefficients excluding the variable ``x``.

        Raises:
            None.

        Side Effects:
            Caches the result after first access.
        """
        return self.args

    @cached_property
    def sols(self) -> List[sym.core.mul.Mul]:
        """Solve the expression set equal to zero.

        Returns analytical solutions, optionally excluding complex values unless
        complex display is enabled. Returns an empty list when solving fails.

        Returns:
            list[sym.core.mul.Mul]: Solutions for the equation ``expr_sym = 0``.

        Raises:
            None.

        Side Effects:
            Caches solutions for reuse.
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
        self, y: Union[int, float, sym.core.expr.Expr] = 0
    ) -> Union[List[sym.core.mul.Mul], sym.core.mul.Mul]:
        """Find solutions when the expression equals a target value.

        Creates a shifted expression by subtracting ``y`` and returns either a list
        of solutions or a single solution depending on availability, reusing
        coefficients if they were previously assigned.

        Args:
            y (int | float | sym.core.expr.Expr): Target value to solve against.

        Returns:
            list[sym.core.mul.Mul] | sym.core.mul.Mul: Solutions of ``expr_sym - y = 0``.

        Raises:
            IndexError: If no solutions exist when attempting to access the first.

        Side Effects:
            None.
        """

        if isinstance(y, unc.core.Variable) or isinstance(y, unc.core.AffineScalarFunc):
            new_expr = FunctionBase1D(self.expr_str + " - p_100")

            list_coefs = list(self.coefs) + [y]
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
    def lambda_fun(self) -> Callable:
        """Generate a numpy-evaluable callable for the expression.

        Returns:
            Callable: Function accepting ``x`` followed by coefficients.

        Raises:
            None.

        Side Effects:
            Caches the generated callable.
        """
        lambda_args = [sym.Symbol("x")] + self.args
        return sym.lambdify(lambda_args, self.expr_sym, "numpy")

    def show_complex(self, show: bool = False) -> "FunctionBase1D":
        """Toggle the display of complex solutions.

        Returns:
            FunctionBase1D: The current instance with updated display preference.

        Raises:
            None.

        Side Effects:
            Flips the `_show_complex` flag controlling solution filtering.
        """
        self._show_complex = show

        return self

    @staticmethod
    def _calculate_uncertainty_analyticaly(
        expr: Union[sym.core.add.Add, sym.core.expr.Expr], parms: List[sym.Symbol] = []
    ) -> sym.core.expr.Expr:
        """Compute propagated uncertainty analytically for an expression.

        Uses partial derivatives with respect to each parameter to accumulate squared
        contributions of parameter uncertainties and returns the square root of their sum.

        Args:
            expr (sym.core.add.Add | sym.core.expr.Expr): Expression whose uncertainty is estimated.
            parms (list[sym.Symbol]): Parameters considered uncertain; defaults to free symbols.

        Returns:
            sym.core.expr.Expr: Symbolic expression representing combined uncertainty.

        Raises:
            None.

        Side Effects:
            None.
        """
        if not parms:
            parms = expr.free_symbols

        unc_list = []
        for parm in parms:
            unc_list.append(
                (sym.diff(expr, parm) * sym.Symbol(f"Delta_{str(parm)}")) ** 2
            )

        return sym.sqrt(np.sum(unc_list))

    def add_coefs(self, coefs: list[Any]) -> "FunctionBase1D":
        """Apply coefficients to the expression and rebuild cached artifacts.

        Substitutes provided coefficients (with or without uncertainties) into the
        symbolic expression, recomputes solutions, and wraps a lambda function that
        binds the coefficients for evaluation.

        Args:
            coefs (list[Any]): Coefficient values matching the detected symbols.

        Returns:
            FunctionBase1D: The current instance with coefficients applied.

        Raises:
            TypeError: If the number of coefficients differs from expectations.

        Side Effects:
            Mutates internal caches and may flag the expression as complex.
        """

        if len(coefs) != len(self.coefs):
            raise TypeError("Number of args is not the same as numer of coefs")

        self.coefs = coefs

        if any(
            [
                isinstance(x, unc.core.Variable)
                or isinstance(x, unc.core.AffineScalarFunc)
                for x in coefs
            ]
        ):
            self._unc_coefs = True

        coefs_dict = {}
        for i in range(len(coefs)):
            coefs_dict[self.args[i]] = coefs[i]

        if self._unc_coefs:
            # separating nominal_values and standart deviations
            nominal_coefs_dict = {}
            std_coefs_dict = {}
            for key in coefs_dict:
                delta = sym.Symbol(f"Delta_{str(key)}")
                if isinstance(coefs_dict[key], unc.core.Variable) or isinstance(
                    coefs_dict[key], unc.core.AffineScalarFunc
                ):
                    nominal_coefs_dict[key] = coefs_dict[key].nominal_value
                    std_coefs_dict[delta] = coefs_dict[key].std_dev
                else:
                    nominal_coefs_dict[key] = coefs_dict[key]
                    std_coefs_dict[delta] = 0

            # making solutions
            sols_list = []
            for sol in self.sols:
                nominal = sol.subs(nominal_coefs_dict)
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
            def wrap_lambdify(func: Callable, *coefs: Any) -> Callable:
                def new_lambda(x: Any) -> Any:
                    return func(x, *coefs)

                return new_lambda

            self.lambda_fun = wrap_lambdify(
                sym.lambdify([sym.Symbol("x")] + self.args, self.expr_sym, "numpy"),
                *coefs,
            )

            # making str expr
            self.expr_unc = self.expr_str
            for key in coefs_dict:
                if isinstance(coefs_dict[key], unc.core.Variable) or isinstance(
                    coefs_dict[key], unc.core.AffineScalarFunc
                ):
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

        Replaces symbolic coefficients with their numeric or uncertainty-aware
        representations and normalizes scientific notation to LaTeX-friendly form.

        Returns:
            str: LaTeX-formatted expression string.

        Raises:
            None.

        Side Effects:
            None.
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

    def _calculate_sols(self, show_unc: Optional[bool] = None) -> list:
        """Prepare solution strings with optional uncertainty notation.

        Recomputes solutions when necessary, formats them for display, and toggles
        inclusion of uncertainty terms depending on the provided flag or internal
        coefficient state.

        Args:
            show_unc (bool | None): Whether to include propagated uncertainties.

        Returns:
            list: Solutions represented as strings or LaTeX fragments.

        Raises:
            None.

        Side Effects:
            May clear cached solutions to force recalculation.
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

    def to_latex_sols(self, y: Any = None, show_unc: bool = False) -> str:
        """Format solutions as a LaTeX array with optional target offset.

        Generates indexed solution strings, optionally subtracting a provided target
        value before solving, and converts scientific notation for readability.

        Args:
            show_unc (bool): Whether to include uncertainties in the output.
            y (Any): Optional target value to subtract prior to solving.

        Returns:
            str: LaTeX-formatted solution array.

        Raises:
            None.

        Side Effects:
            May trigger recalculation of cached solutions.
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

        Differentiates the expression with respect to ``x`` the specified number of
        times and propagates existing coefficients to the new function if present.

        Args:
            level (int): Order of the derivative to compute.

        Returns:
            FunctionBase1D: New instance representing the derivative.

        Raises:
            None.

        Side Effects:
            None.
        """
        expr = self.expr_sym

        deriv = sym.diff(expr, sym.Symbol("x"), level)

        new_expr_str = str(deriv)

        new_func = FunctionBase1D(new_expr_str)

        if self._added_coefs:
            new_func.add_coefs(self.coefs)

        return new_func


class Poly(FunctionBase1D):
    def __init__(self, degree: int = str) -> None:
        """Create a polynomial expression of the specified degree.

        Args:
            degree (int | type[str]): Polynomial degree; defaults to a placeholder `str`.

        Returns:
            None

        Raises:
            None.

        Side Effects:
            Initializes the base class with a generated polynomial expression string.
        """
        self.degree = degree

        expr_str = ""
        for i in range(degree):
            expr_str += f"p_{i}*x**{degree - i} + "
        expr_str += f"p_{degree}"

        super().__init__(expr_str)
        # self.expr_str = expr_str

    @cached_property
    def sols(self) -> List[sym.core.mul.Mul]:
        """Return analytical solutions for the polynomial when solvable.

        Returns:
            list[sym.core.mul.Mul]: Solutions for the polynomial equation.

        Raises:
            TypeError: If the polynomial degree exceeds four and cannot be solved analytically.

        Side Effects:
            None.
        """
        if self.degree > 4:
            raise TypeError(
                "Cant find analytical solution for Poly with degree greater than 4"
            )
        else:
            return super().sols


class Hyper(FunctionBase1D):
    def __init__(self, style: int = 0) -> None:
        """Initialize a hyperbolic expression variant.

        Args:
            style (int): Selector for expression form; ``0`` uses ``p_0*x / (x + p_1)``,
                otherwise uses ``p_0 / (x + p_1) + p_2``.

        Returns:
            None

        Raises:
            None.

        Side Effects:
            Sets up the base class with the chosen expression.
        """
        if style == 0:
            expr_str = "p_0*x / (x + p_1)"
        else:
            expr_str = "p_0 / (x + p_1) + p_2"

        super().__init__(expr_str)
