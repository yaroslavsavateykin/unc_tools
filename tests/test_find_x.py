import numpy as np
import uncertainties as unc

import sys
import os

sys.path.append(os.path.expanduser("~/unc_tools/"))

from unc_tools import UncRegression

x = np.linspace(0, 10, 100) + np.random.uniform(low=-0.05, high=0.05, size=100)
y = 4 * np.linspace(0, 10, 100) + 3 + np.random.uniform(low=-0.05, high=0.05, size=100)

reg = UncRegression(x, y)

y0 = 25
x0 = 5
x0_1 = reg.find_x(y0)
x0_2 = reg.find_x(y0, x0=x0)
print(f"{x0_1:.10f}, {x0_2:.10f}")


y0 = unc.ufloat(25, 0.1)
x0 = 5
x0_1 = reg.find_x(y0)
x0_2 = reg.find_x(y0, x0=x0)
print(f"{x0_1:.10f}, {x0_2:.10f}")

y0 = unc.ufloat(25, 0.1)
x0 = unc.ufloat(5, 0.1)
x0_1 = reg.find_x(y0)
x0_2 = reg.find_x(y0, x0=x0)
print(f"{x0_1:.10f}, {x0_2:.10f}")
