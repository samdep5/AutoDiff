from team48_autodiff_package.dualNumber.dual_numbers import *
import numpy as np
from numbers import Number

def derivative(f, p, values):
    '''
    Find the derivative of a function `f` at point `values` with respect to a seed vector `p`

    Parameters
    ----------
    f :
        A function (number(s) -> number) made up of the math functions compatible with DualNumbers
    p :
        A seed vector (array of numbers) i.e. [1,0,0]
    values :
        The inputs to the function (array of numbers of length equal to parameters of `f`)
    
    Returns
    -------
    Derivative of `f` (Number which is the dual component of the output) 
    I think what happens when you put in a vector with more than 1 value it 
    adds together all the derivatives on each step.
    '''
    args = []
    if isinstance(p, (list,np.ndarray)):
        for i, val in zip(p, values):
            args.append(DualNumber(val, i))
    elif isinstance(p, Number):
        # Should take a number and then assign dual number via that vector
        for i, val in enumerate(values):
            if i == p:
                args.append(DualNumber(val, 1))
            else:
                args.append(DualNumber(val, 0))
    else:
        raise ValueError("Seed vector must be a list, numpy array, or number")
    # Handle function input as string. 
    # String evaluation works only for comprehensions and expressions
    # All vars in values vector must be used in function string
    if isinstance(f,str):
        # https://realpython.com/python-eval-function/
        # allow use of DualNumber functions
        allowed_names = {"exp": exp, "sin": sin, "cos": cos, "tan": tan, "log": log, "sqrt": sqrt}
        code = compile(f, "<string>", "eval")
        # populate namespace with function vars
        vars = code.co_varnames
        for i,v in enumerate(args):
            allowed_names[vars(i)] = v
        # check to make sure they didn't use any disallowed functions
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Can't use '{name}'. Use only functions defined for dual numbers.")
        # eveluate expession, setting __builtins__ to {} to avoid malicious code injection
        return eval(code, {"__builtins__": {}}, allowed_names).dual
    else:
        return f(*args).dual

def gradient(f, values):
    '''
    Find the gradient of a function at a given point

    Parameters
    ----------
    f :
        A function or vector of functions
    values :
        A number or array of numbers

    Returns
    -------
    The gradient of the function at point `values`
    '''
    # If there are multiple variables, calculate the derivative with respect to each:
    if isinstance(f, list) and isinstance(values, list):
        grad = np.ndarray([])
        for i, g in enumerate(f):
            a = [derivative(g, v, values) for v in values]
            print(a)
            grad = np.append(grad, a)
        return grad
    # If there are mutliple functions but only one variable:
    elif isinstance(f, list):
        grad = np.ndarray([])
        for i, g in enumerate(f):
            grad.append(derivative(g, 0, values))
        return grad
    # If there is only one function and multiple variables:
    elif isinstance(values, list):
        return np.ndarray([derivative(f, v, values) for v in values])
    # If there is only one function and one variable:
    else:
        return np.ndarray([derivative(f, 0, values)])

def derivative_approximation(fx, values, p):
    '''
    Approximates the derivative of `fx` at `values` with respect to `p`

    Parameters
    ----------
    fx :
        Function with numerical output
    values :
        inputs to the `fx`
    
    Returns
    -------
    Derivative approximation of d(fx(values))/dx
    '''
    h = 1.e-8
    out = fx(*values)
    values[p] += h
    nudge = fx(*values)
    approx = (nudge - values)/ h
    return approx

def check_error(test_value, approx):
    '''
    Makes sure difference between inputs is within acceptible margin of error

    Parameters
    ----------
    test_value :
        A number representing the output of a function
    approx :
        An approximation of the test value
    
    Returns
    -------
    True if the difference is within accepted error margin, False if otherwise
    '''
    err = 1.e-6
    accepted_error = err*(1 + abs(approx))
    return accepted_error > abs(approx - test_value)