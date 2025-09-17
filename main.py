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
    from unc_tools.default_functions import Poly

    a2 = Poly(2).show_complex().add_coefs([4, unc.ufloat(2, 0.001), 2])

    print(a2.to_latex_sols())


@time
def main2():
    from unc_tools.default_functions import Hyper

    a = Hyper(style=0).add_coefs([4, 2])

    print(a.to_latex_sols())


if __name__ == "__main__":
    main2()
