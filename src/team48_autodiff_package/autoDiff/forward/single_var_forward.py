from team48_autodiff_package.dualNumber.dual_numbers import *
from numbers import Number

def single_derivative_approximation(fx, x):
    '''Return the derivative approximation for the function `fx` at `x`.'''
    h = 1.e-8
    approx = (fx(x + h) - fx(x))/ h
    return approx

def check_error(test_value, approx):
    '''Return True if error between inputs is trivial else False.'''
    err = 1.e-6
    accepted_error = err*(1 + abs(approx))
    return accepted_error > abs(approx - test_value)

def single_derivative(fx, input):
    '''
    Return derivative calculation for the function `fx` at `x`. 

    String evaluation works only for comprehensions and expressions
    (no if, while, def, etc.) but can handle all of the functions in 
    the DualNumber class.

    Parameters
    ----------
    fx :
        A function made up of the DualNumber compatible operations
    x : 
        A numerical input to `fx`

    Returns
    -------
    The derivative of fx at x
    '''
    if isinstance(fx,str):
        code = compile(fx, "<string>", "eval")
        # small bug if the variable is ever called input it will fail
        v = code.co_names[0]
        locals()[v] = DualNumber(2,1)
        return eval(fx).dual
    out = fx(DualNumber(input))
    if isinstance(out, DualNumber):
        return out.dual
    else:
        return out

