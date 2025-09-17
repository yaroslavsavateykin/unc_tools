from ..unc_tools.unc_regression import UncRegression
import uncertainties as unc
import numpy as np
import scipy.optimize as opt


def f(x, a, b):
    return np.sin(x) * a + b


x = np.linspace(0, 10, 10)
y = np.sin(x) * 4 + 3 + np.random.normal(0, 0.5, 10)
y = 5 * x + 4 + np.random.normal(0, 0.001, 10)

uf = unc.ufloat(0, 0.001)
print(f"Заданное значение y в точке: {uf}")

print("-" * 50)

reg = UncRegression(x, y)
print(f"Коэффициенты фита: {reg.coefs}")

a = reg.find_x(y=uf, x0=0)
print(f"Численное решение с погрешностью: {a}")

b = reg.predict_with_uncertainty(np.array([a]))
print(f"Найденное значение y в точке: {b}")

print("-" * 50)

res = opt.root_scalar(f=f, x0=0, args=tuple(reg.coefs_values))
print(f"Коэффициенты фита: {reg.coefs_values}")
print(f"Численное решение без погрешности: {res.root}")

f = f(res.root, *tuple(reg.coefs_values))
print(f"Найденное значение y в точке: {f}")
