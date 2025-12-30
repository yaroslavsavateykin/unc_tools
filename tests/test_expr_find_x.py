import numpy as np
import uncertainties as unc

import sys
import os

sys.path.append(os.path.expanduser("~/unc_tools/"))

from unc_tools import UncRegression, Poly

x = np.linspace(0,10,100) + np.random.uniform(low=-.05, high=.05, size= 100)
y = 5 * np.linspace(0,10,100) + 3 + np.random.uniform(low=-.05, high=.05, size= 100)

reg = UncRegression(x,y)

expr = reg.expression

y0 = 5
x0 = expr.find_sols(y0)

print(x0, y0)

y0 = unc.ufloat(5,0.5)
x0 = expr.find_sols(y0)

print(x0, y0)
