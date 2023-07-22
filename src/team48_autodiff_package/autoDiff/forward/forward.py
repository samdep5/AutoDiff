from team48_autodiff_package.dualNumber.dual_numbers import *
import numpy as np
from numbers import Number

# To get derivative of something wrt to a single variable we just set all the rest of the dual components to 0 and leave only 1

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
        # should take a number and then assign dual number via that vector
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
        # far less safe than what we had before
        code = compile(f, "<string>", "eval")
        variables = sorted([x for x in list(code.co_names) if len(x) == 1])
        for i,v in enumerate(variables):
            # cannot be p or f
            globals()[v] = args[i]
        return eval(f).dual

        # # This was really cool so we are leaving it, but for now we are going to use another method to handle strings
        # # https://realpython.com/python-eval-function/
        # # allow use of DualNumber functions
        # allowed_names = {"exp": exp, "sin": sin, "cos": cos, "tan": tan, "log": log, "sqrt": sqrt}
        # code = compile(f, "<string>", "eval")
        # # populate namespace with function vars
        # vars = code.co_varnames
        # for i,v in enumerate(args):
        #     allowed_names[vars(i)] = v
        # # check to make sure they didn't use any disallowed functions
        # for name in code.co_names:
        #     if name not in allowed_names:
        #         raise NameError(f"Can't use '{name}'. Use only functions defined for dual numbers.")
        # # eveluate expession, setting __builtins__ to {} to avoid malicious code injection
        # return eval(code, {"__builtins__": {}}, allowed_names).dual
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
    grad = []
    # Check if user gave one or multiple dimensions
    if isinstance(f,(list, np.ndarray)):
        # For each dimension
        for g in f:
            arr = []
            # check if user gave one or multiple inputs
            if isinstance(values,list):
                # if multiple, find derivative of each
                for i in range(len(values)):
                    arr.append(derivative(g, i, values))
                grad.append(arr)
                # if one value, derive it (single_derivative?)
            else: 
                arr.append(derivative(g, 0, values))
                grad.append(arr)
    else:
        arr = []
        for i in range(len(values)):
            arr.append(derivative(f, i, values))
        grad = arr
    return grad

def derivative_approximation(fx, values, p):
    '''
    Approximates the derivative of `fx` at `values` using `p`

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
    def single_approx(p, fx):
        if isinstance(p,(list, np.ndarray)):
            store = 0
            for n,v in enumerate(p):
                out = fx(*values)
                values[n] += h
                nudge = fx(*values)
                approx = (nudge - out)/ h
                store += v*approx
            return store
        else:
            out = fx(*values)
            values[p] += h
            nudge = fx(*values)
            approx = (nudge - out)/ h
            return approx
    
    if isinstance(fx, list):
        ret = []
        for f in fx:
            ret.append(single_approx(p,f))
        return ret
    else:
        return single_approx(p,fx)

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


