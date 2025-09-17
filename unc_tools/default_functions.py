from ast import Call, Dict, Raise, expr
from functools import cached_property
from os import replace
from typing import Callable, List, Union
from matplotlib.pyplot import show
import sympy as sym
from sympy.matrices.expressions.matmul import new
import uncertainties as unc
import numpy as np


class FunctionBase1D:
    def __init__(self, expr_str: str = ""):
        self.expr_str = expr_str
        self.expr_unc: str = ""
        self.args: list = []
        self._added_coefs = False
        self._unc_coefs = False
        self._complex = False
        self._show_complex = False

        # self.expr: Union[sym.core.add.Add, sym.core.expr.Expr]
        # self.coefs: List[sym.core.symbol.Symbol]
        # self.sols = List[sym.core.mul.Mul]
        # self.lambda_fun: Callable

    @cached_property
    def expr(self) -> Union[sym.core.add.Add, sym.core.expr.Expr]:
        return sym.parse_expr(self.expr_str)

    @cached_property
    def coefs(self) -> List[sym.core.symbol.Symbol]:
        return sorted(
            [x for x in list(self.expr.free_symbols) if not str(x) == "x"],
            key=lambda s: str(s),
        )

    @cached_property
    def sols(self) -> List[sym.core.mul.Mul]:
        solution = sym.solvers.solve(self.expr, sym.Symbol("x"))

        if not solution:
            raise TypeError("Cant find analytical solution for given expression")

        if self._show_complex:
            return solution
        else:
            return [x for x in solution if not x.is_complex]

    @cached_property
    def lambda_fun(self) -> Callable:
        lambda_args = [sym.Symbol("x")] + self.coefs
        return sym.lambdify(lambda_args, self.expr, "numpy")

    def show_complex(self):
        self._show_complex = not self._show_complex
        return self

    @staticmethod
    def _calculate_uncertainty_analyticaly(
        expr: Union[sym.core.add.Add, sym.core.expr.Expr],
    ):
        parms = expr.free_symbols

        unc_list = []
        for parm in parms:
            unc_list.append(
                (sym.diff(expr, parm) * sym.Symbol(f"Delta_{str(parm)}")) ** 2
            )

        return sym.sqrt(np.sum(unc_list))

    def add_coefs(self, args: list):
        self.args = args

        if len(args) != len(self.coefs):
            raise TypeError("Number of args is not the same as numer of coefs")

        if any([isinstance(x, unc.core.Variable) for x in args]):
            self._unc_coefs = True

        args_dict = {}
        for i in range(len(args)):
            args_dict[self.coefs[i]] = args[i]

        if self._unc_coefs:
            # separating nominal_values and standart deviations
            nominal_args_dict = {}
            std_args_dict = {}
            for key in args_dict:
                delta = sym.Symbol(f"Delta_{str(key)}")
                if isinstance(args_dict[key], unc.core.Variable):
                    nominal_args_dict[key] = args_dict[key].nominal_value
                    std_args_dict[delta] = args_dict[key].std_dev
                else:
                    nominal_args_dict[key] = args_dict[key]
                    std_args_dict[delta] = 0

            # making solutions
            sols_list = []
            for sol in self.sols:
                nominal = sol.subs(nominal_args_dict)
                if nominal.is_complex:
                    self._complex = True

                    if self._show_complex:
                        std = self._calculate_uncertainty_analyticaly(sol).subs(
                            {**nominal_args_dict, **std_args_dict}
                        )
                        real_part = unc.ufloat(sym.re(nominal), abs(sym.re(std)))
                        im_part = unc.ufloat(sym.im(nominal), abs(sym.im(std)))
                        sols_list.append((real_part, im_part))
                else:
                    std = self._calculate_uncertainty_analyticaly(sol).subs(
                        {**nominal_args_dict, **std_args_dict}
                    )
                    sols_list.append(unc.ufloat(nominal, std))

            self.sols = sols_list

            # making lambda function
            def wrap_lambdify(func: Callable, *args) -> Callable:
                def new_lambda(x):
                    return func(x, *args)

                return new_lambda

            self.lambda_fun = wrap_lambdify(
                sym.lambdify(sym.Symbol("x"), self.expr, "numpy"), args
            )

            # making str expr
            self.expr_unc = self.expr_str
            for key in args_dict:
                if isinstance(args_dict[key], unc.core.Variable):
                    replacement = f"unc.ufloat({args_dict[key].nominal_value},{args_dict[key].std_dev})"
                else:
                    replacement = str(args_dict[key])

                self.expr_unc = str(self.expr_unc).replace(str(key), replacement)

            self._added_coefs = True

        else:
            self.expr = self.expr.subs(args_dict)
            self.lambda_fun = sym.lambdify(sym.Symbol("x"), self.expr, "numpy")
            self._added_coefs = True

            self.sols = [x.subs(args_dict) for x in self.sols]
            if any([x.is_complex for x in self.sols]):
                self._complex = True

        return self

    def to_latex_expr(self):
        latex = sym.latex(self.expr)

        if self._added_coefs:
            for i in range(len(self.coefs)):
                if isinstance(self.args[i], unc.core.Variable):
                    replacement = f"({self.args[i]})"
                else:
                    replacement = f"{self.args[i]}"

                latex = latex.replace(sym.latex(self.coefs[i]), replacement)

            latex = latex.replace("+/-", r" \\pm ")

        return latex

    def to_latex_sols(self, show_unc=None):
        if not show_unc:
            show_unc = self._unc_coefs

        latex_sols = []

        if self._added_coefs:
            if self._show_complex:
                del self.__dict__["sols"]
                self.add_coefs(self.args)
                latex_sols = [
                    str(x) if not isinstance(x, tuple) else f"({x[0]}) + ({x[1]}) i"
                    for x in self.sols
                ]
            else:
                latex_sols = self.sols

            latex_sols = [f"{x}".replace("+/-", " \\pm ") for x in latex_sols]

        else:
            del self.__dict__["sols"]
            self.sols
            latex_sols = [sym.latex(x) for x in self.sols]

        for i, sol in enumerate(latex_sols):
            latex_sols[i] = f"x_{i + 1} = {sol} \\\\"

        lines = f"\\begin{{split}}\n{'\n'.join(latex_sols)}\n\\end{{split}}"

        return lines


class Poly(FunctionBase1D):
    def __init__(self, degree: int = str):
        super().__init__()

        self.degree = degree

        expr_str = ""
        for i in range(degree):
            expr_str += f"p_{i}*x**{degree - i} + "
        expr_str += f"p_{degree}"

        self.expr_str = expr_str

    @cached_property
    def sols(self):
        if self.degree > 4:
            raise TypeError(
                "Cant find analytical solution for Poly with degree greater than 4"
            )
        else:
            return super().sols


class Hyper(FunctionBase1D):
    def __init__(self, style=0):
        super().__init__()

        if style == 0:
            self.expr_str = "p_0*x / (x + p_1)"
        else:
            self.expr_str = "p_0 / (x + p_1) + p_2"
