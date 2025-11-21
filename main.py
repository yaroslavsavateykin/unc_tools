import uncertainties as unc
import sympy as sym


def time(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()

        print(f"Time used on {str(func)}: {(end - start):.2f} seconds")
        return r

    return wrapper


@time
def main1():
    from unc_tools.default_functions import Poly, FunctionBase1D

    a = FunctionBase1D(expr_str="1 / (a-b) * log((b-x)/(a-x))")
    print(a.coefs)
    print(a.to_latex_expr())


@time
def main2():
    from .unc_tools.default_functions import Poly

    a = Poly(2)  # .show_complex()

    a = a.add_coefs([unc.ufloat(4, 0.001), unc.ufloat(2, 0.001), 3])

    print(a.find_sols(y=1))

    print(a.to_latex_sols(show_unc=False, y=1))


def main3():
    import unc_tools
    import sympy as sym
    import numpy as np
    import unc_tools.patches

    expr = sym.parse_expr("a + b + c")

    a, b = sym.symbols("a b")

    expr = expr.subs({a: unc.ufloat(5, 0.1)})

    print(expr.is_unc)
    expr = expr.subs({b: unc.ufloat(5, 0.1)})
    print(expr.is_unc)

    fun1 = sym.lambdify(sym.symbols("a"), expr, "unc")

    b = [
        unc.ufloat(100, 0.001),
        unc.ufloat(100, 0.002),
    ]
    b = np.asarray(b)

    y = fun1(b, b)

    print(y)


def main4():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from unc_tools import UncRegression
    from uncertainties.umath import exp
    from uncertainties import unumpy as unp

    df = pd.DataFrame(
        {
            "k": [
                unc.ufloat(0.09120, 0.0001),
                unc.ufloat(0.1581, 0.0001),
                unc.ufloat(0.2713, 0.0002),
            ],
            "temp": [295.95, 302.95, 312.75],
        }
    )

    reg = UncRegression(1 / df.temp, unp.log(df.k))

    deriv = reg.expression.deriv()

    print(deriv.expr_str)

    # reg.to_csv("test.csv", index=False)
    # print(reg.to_df())


def main5():
    import numpy as np
    from unc_tools import UncRegression, FunctionBase1D
    import matplotlib.pyplot as plt

    t = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 150, 200, 300, 500]
    )
    P_exp = np.array(
        [
            0.00,
            0.51,
            0.99,
            1.43,
            1.84,
            2.22,
            2.58,
            2.91,
            3.22,
            3.51,
            3.78,
            5.69,
            6.75,
            7.40,
            7.86,
            9.16,
            9.88,
            10.32,
            10.75,
            10.97,
        ]
    )

    func = FunctionBase1D("A1*(1-exp(-k1*x)) + A2*(1-exp(-k2*x))")

    reg = UncRegression(t, P_exp, func=func)
    reg.plot()

    func1 = FunctionBase1D("A1*(1-exp(-k1*x))").add_coefs([reg.coefs[0], reg.coefs[2]])
    func2 = FunctionBase1D("A1*(1-exp(-k1*x))").add_coefs([reg.coefs[1], reg.coefs[3]])

    plt.plot(t, func1.lambda_fun(t))

    plt.savefig("test.png")


if __name__ == "__main__":
    main3()
