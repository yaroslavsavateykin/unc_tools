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
    from unc_tools.default_functions import Hyper

    a = Hyper(style=0).add_coefs([4, 2])

    print(a.to_latex_sols(show_unc=True))


if __name__ == "__main__":
    main1()
