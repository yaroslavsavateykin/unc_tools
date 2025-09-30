import uncertainties as unc
import sympy as sym

from unc_tools.unc_regression import UncRegression


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

    # print(a.coefs)
    print(a.to_latex_expr())


@time
def main2():
    from unc_tools.default_functions import Poly

    a = Poly(2)  # .show_complex()

    a = a.add_coefs([unc.ufloat(4, 0.001), unc.ufloat(2, 0.001), 3])

    # print(a.find_sols(y=1))

    print(a.to_latex_sols(show_unc=False, y=1))


def main3():
    from unc_tools.default_functions import Poly, FunctionBase1D

    print(issubclass(type(Poly(3)), FunctionBase1D))


def main4():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from unc_tools import UncRegression
    from uncertainties.umath import exp

    df = pd.DataFrame(
        {"k": [0.09120, 0.1581, 0.2713], "temp": [295.95, 302.95, 312.75]}
    )

    reg = UncRegression(1 / df.temp, np.log(df.k))
    fig, ax = plt.subplots(figsize=(10, 5))
    print(type(ax))
    y = -1.6
    x = reg.find_x(y)
    print(x)

    # ax.plot(x, y)

    reg.plot(ax=ax, show_band=True)
    # fig.show()
    # fig.savefig("final.png")


if __name__ == "__main__":
    main4()
