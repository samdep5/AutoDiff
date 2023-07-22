import numpy as np
# not sure the imprt below is needed anymore?
from team48_autodiff_package.autoDiff.forward.single_var_forward import *
from team48_autodiff_package.autoDiff.forward.forward import *
from team48_autodiff_package.dualNumber.dual_numbers import *

# Note: entering an equation directly with dual 
# number vars will immediately yield derivative without this class.

'''
Automatic Differentiation class object.

Initialized with a function value and can
be called repeatedly to evaluate the 
function at different points.
'''
class AutoDiff():
    '''
    Automatic Differentiation class object.

    Initialized with a function value and can
    be called repeatedly to evaluate the
    function at different points.

    Attributes
    ----------
    func :
    the associated function which must be dual-number compatible

    Methods
    -------
    __init__(func):
        create new AutoDiff object with function `func`
    __call__(x, p=[1]):
        return the derivative of the function with input(s) x, weighted by seed vector p
    gradient:
        return the gradient of the fucntion with input(s) x
    evaluate:
        return the value of the function with input(s) x
    '''

    def __init__(self, func):
        self.func = func

    # On call, return the derivative of the function
    # Seed vector set to 1 for first var and 0 for all others by default
    def __call__(self, x, p=[1]):
        '''Return the derivative of the function.'''
        # If they sent multiple variables, evaluate appropriately
        if isinstance(x, (list, np.ndarray)) and not isinstance(p, Number):
            return np.dot(gradient(self.func, x), p)
        elif isinstance(x, (list, np.ndarray)):
            return derivative(self.func, p, x)
        elif isinstance(x, (int, float, DualNumber)):
            return single_derivative(self.func, x)
        else:
            raise TypeError("Please enter your input(s) as and integer, float, or a list or Numpy array thereof")

    def gradient(self, x):
        '''Return the gradient of the function at `x`'''
        return gradient(self.func, x)

    def evaluate(self, x):
        '''Evaluate the function at `x`'''
        if isinstance (x, (tuple, np.ndarray, list)) and not isinstance(self.func, str):
            return self.func(*x)
        elif not isinstance(self.func, str):
            return self.func(x)
        else:
            return eval(self.func)