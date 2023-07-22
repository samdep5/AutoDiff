from team48_autodiff_package.autoDiff import *
from numbers import Number

def approx(fx, values):
    h = 1.e-8
    approx = []
    # if isinstance(values, Number):
    h = 1.e-8
    approx = (fx(values+ h) - fx(values))/ h
    return approx
    # else:
    #     for i in range(len(values)):
    #         shifted_args = values[:]
    #         shifted_args[i] = shifted_args[i] + h
    #         approx.append((fx(*shifted_args) - fx(*values)) / h)
    #     return approx


