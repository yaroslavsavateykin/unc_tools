from ast import Call, Dict, Raise, expr
from functools import cached_property
from math import exp
from os import replace
from typing import Callable, List, Union
from matplotlib.pyplot import show
import sympy as sym
from sympy.matrices.expressions.matmul import new
import uncertainties as unc
import numpy as np
import re


class FunctionBase1D:
    def __init__(self, expr_str: str = ""):
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
        return self.args

    @cached_property
    def sols(self):
        try:
            solution = sym.solvers.solve(self.expr_sym - 0, sym.Symbol("x"))

            # if not solution:
            #    raise TypeError("Cant find analytical solution for given expression")

            if self._show_complex:
                return solution
            else:
                solution = [x for x in solution if not x.is_imaginary]
                return solution
        except:
            return []

    def find_sols(self, y=0) -> List[sym.core.mul.Mul]:
        # self.expr -= y

        new_expr = FunctionBase1D(self.expr_str)
        new_expr.expr -= y
        # coefs = self.args + [y]
        # print(coefs)

        if self._added_coefs:
            new_expr.add_coefs(self.coefs)

        if len(new_expr.sols) > 1 and hasattr(new_expr.sols, "__iter__"):
            return new_expr.sols
        else:
            return new_expr.sols[0]

    @cached_property
    def lambda_fun(self) -> Callable:
        lambda_args = [sym.Symbol("x")] + self.args
        return sym.lambdify(lambda_args, self.expr_sym, "numpy")

    def show_complex(self):
        self._show_complex = not self._show_complex
        return self

    @staticmethod
    def _calculate_uncertainty_analyticaly(
        expr: Union[sym.core.add.Add, sym.core.expr.Expr], parms=[]
    ):
        if not parms:
            parms = expr.free_symbols

        unc_list = []
        for parm in parms:
            unc_list.append(
                (sym.diff(expr, parm) * sym.Symbol(f"Delta_{str(parm)}")) ** 2
            )

        return sym.sqrt(np.sum(unc_list))

    def add_coefs(self, coefs: list):
        self.coefs = coefs

        if len(coefs) != len(self.coefs):
            raise TypeError("Number of args is not the same as numer of coefs")

        if any([isinstance(x, unc.core.Variable) for x in coefs]):
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
                if isinstance(coefs_dict[key], unc.core.Variable):
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
            def wrap_lambdify(func: Callable, *coefs) -> Callable:
                def new_lambda(x):
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

    def to_latex_expr(self):
        latex = sym.latex(self.expr_sym)

        if self._added_coefs:
            for i in range(len(self.coefs)):
                if isinstance(self.coefs[i], unc.core.Variable):
                    replacement = f"({self.coefs[i]})"
                else:
                    replacement = f"{self.coefs[i]}"

                latex = latex.replace(sym.latex(self.args[i]), replacement)

            latex = "$y = " + latex.replace("+/-", r" \pm ") + "$"
        latex = re.sub(
            r"e([+-])(\d+)",
            lambda m: rf"\cdot 10^{{{m.group(1)}{str(int(m.group(2)))}}}",
            latex,
        )
        return rf"{latex}"

    def _calculate_sols(self, show_unc=None):
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

    def to_latex_sols(self, show_unc=True, y=None):
        show_unc = show_unc if not self._unc_coefs else self._unc_coefs

        if not y:
            latex_sols = self._calculate_sols(show_unc=show_unc)
        else:
            new_expr = FunctionBase1D(self.expr_str + " - A100")
            new_expr.add_coefs(self.coefs + [y])
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

    def deriv(self, level=1):
        expr = self.expr_sym

        deriv = sym.diff(expr, sym.Symbol("x"), level)

        new_expr_str = str(deriv)

        new_func = FunctionBase1D(new_expr_str)

        if any(self.coefs):
            new_func.add_coefs(self.coefs)

        return new_func


class Poly(FunctionBase1D):
    def __init__(self, degree: int = str):
        self.degree = degree

        expr_str = ""
        for i in range(degree):
            expr_str += f"p_{i}*x**{degree - i} + "
        expr_str += f"p_{degree}"

        super().__init__(expr_str)
        # self.expr_str = expr_str

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
        if style == 0:
            expr_str = "p_0*x / (x + p_1)"
        else:
            expr_str = "p_0 / (x + p_1) + p_2"

        super().__init__(expr_str)
