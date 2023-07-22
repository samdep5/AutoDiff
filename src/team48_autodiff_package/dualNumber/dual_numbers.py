from numbers import Number
import math
import numpy as np

class DualNumber:
    """
   Dual number type for automatic differentiation.
 
   Attributes
   ----------
   real :
       the real component, representing an intermediate value of a variable at a given point
   dual :
       the dual component, representing the derivative of the variable at a given point
   """


    def __init__(self, real, dual=1):
        """Initialize DualNumber with realNum and dualNum (default=1) parts."""
        self.real = real
        self.dual = dual

    def __eq__(self, other):
        """Overloaded function to compare DualNumber to another DualNumber."""
        if isinstance(other, DualNumber):
            return '{:g}'.format(float('{:.{p}g}'.format(other.dual, p=10))) == '{:g}'.format(float('{:.{p}g}'.format(self.dual, p=10))) and '{:g}'.format(float('{:.{p}g}'.format(other.real, p=10))) == '{:g}'.format(float('{:.{p}g}'.format(self.real, p=10)))
        else:
            raise TypeError("Values must be of type int, float, or dual number")
    
    def _add(self, other):
        """Private function to add DualNumber with scalar or DualNumber"""
        if isinstance(other, DualNumber):
            realVal = self.real + other.real
            dualVal = self.dual + other.dual
            return DualNumber(realVal, dualVal)
        elif isinstance(other, float) or isinstance(other, int):
            realVal = self.real + other
            dualVal = self.dual
            return DualNumber(realVal, dualVal)
        else:
            raise TypeError("Values must be of type int, float, or dual number")
        
    def _mul(self, other):
        """Private function to multiply DualNumber with scalar or DualNumber."""
        if isinstance(other, DualNumber):
            realVal = self.real * other.real
            dualVal = (self.real * other.dual) + (self.dual * other.real)
            return DualNumber(realVal, dualVal)
        elif isinstance(other, float) or isinstance(other, int):
            realVal = self.real * other
            dualVal = self.dual * other
            return DualNumber(realVal, dualVal)
        else:
            raise TypeError("Values must be of type int, float, or dual number")
    
    def _sub(self, other, ordered = True):
        """Private function to subtract scalar or DualNumber from DualNumber."""
        if isinstance(other, DualNumber) and ordered:
            realVal = self.real - other.real
            dualVal = self.dual - other.dual
            return DualNumber(realVal, dualVal)
        elif isinstance(other, Number) and ordered:
            realVal = self.real - other
            dualVal = self.dual
            return DualNumber(realVal, dualVal)
        elif not ordered and isinstance(other, Number):
            return DualNumber(other - self.real, -1*self.dual)
        else:
            raise TypeError("Values must be of type int, float, or dual number")

    def _div(self, other, ordered = True):
        """Private function to divide DualNumber by scalar or DualNumber."""
        if isinstance(other, DualNumber) and ordered:
            if other.real == 0:
                raise ZeroDivisionError
            realVal = self.real / other.real
            dualVal = ((self.dual * other.real) - (self.real * other.dual)) / (other.real**2)
        elif isinstance(other, Number) and ordered:
            if other == 0:
                raise ZeroDivisionError
            realVal = self.real / other
            dualVal = self.dual / other
        elif isinstance(self, DualNumber) and not ordered:
            if self.real == 0:
                raise ZeroDivisionError
            realVal = other /self.real 
            dualVal = (-other.real * self.dual) / (self.real**2)
        else:
            raise TypeError("Values must be of type int, float, or dual number")
        return DualNumber(realVal, dualVal)
    
    def _pow(self, other, ordered = True):
        """Private function to raise Dual number to a power or raise number to dual number"""
        if ordered and isinstance(other, DualNumber):
            realVal = self.real**other.real
            dualVal = self.real**other.real * ((other.dual * np.log(self.real)) + (self.dual * other.real / self.real))
            newNum = DualNumber(realVal, dualVal)
        elif ordered and isinstance(other, Number): 
            realVal = self.real**other
            dualVal = self.real**other*(self.dual * other.real / self.real)
            newNum = DualNumber(realVal, dualVal)
        elif not ordered and isinstance(other, Number):
            realVal = other**self.real
            dualVal = (other**self.real) * self.dual * np.log(other)
            newNum = DualNumber(realVal, dualVal)
        else:
            raise TypeError("Values must be of type int, float, or dual number")
        return newNum

    def _neg(self):
        """Private function to negate a DualNumber"""
        return self * -1

    def __neg__(self):
        """Overloaded function to negate DualNumbers"""
        return self._neg()

    def __add__(self, other):
        """Overloaded function to add DualNumber with scalar or DualNumber."""
        return self._add(other)
    
    def __radd__(self, other):
        """Overloaded function to add scalar or DualNumber with a DualNumber."""
        return self._add(other)
    
    def __mul__(self, other):
        """Overloaded function to multiply DualNumber with scalar or DualNumber."""
        return self._mul(other)
    
    def __rmul__(self, other):
        """Overloaded function to multiply scalar or DualNumber with a DualNumber."""
        return self._mul(other)
    
    def __sub__(self, other):
        """Overloaded function to subtract scalar or DualNumber from DualNumber."""
        return self._sub(other)

    def __rsub__(self, other):
        """Overloaded function to subtract DualNumber from scalar or DualNumber."""
        return self._sub(other, ordered = False)

    def __truediv__(self, other):
        """Overloaded function to divide DualNumber by scalar or DualNumber."""
        return self._div(other)
    
    def __rtruediv__(self, other):
        """Overloaded function to divide scalar or DualNumber by DualNumber."""
        return self._div(other, ordered = False)
    
    def __pow__(self, other):
        """Overloaded function to raise DualNumber to a DualNumber or a scalar."""
        return self._pow(other)
    def __rpow__(self, other):
        """Overloaded function to raise DualNumber or a scalar to a DualNumber."""
        return self._pow(other, False)

def exp(x):
    '''Calculate e (~2.718282) raised to a DualPower.'''
    if isinstance(x, DualNumber):
        return np.e**x
    else:
        return np.exp(x)
    

def sin(x):
    '''Calculate the sin of a DualNumber.'''
    if isinstance(x, DualNumber):
        realVal = np.sin(x.real)
        dualVal = x.dual * np.cos(x.real)
        return DualNumber(realVal, dualVal)
    else:
        return np.sin(x)

def cos(x):
    '''Calculate the cos of a DualNumber.'''
    if isinstance(x, DualNumber):
        realVal = np.cos(x.real)
        dualVal = -x.dual * np.sin(x.real)
        return DualNumber(realVal, dualVal)
    else:
        return np.cos(x)

def tan(x):
    '''Calculate the tan of a DualNumber.'''
    if isinstance(x,DualNumber):
        realVal = np.tan(x.real)
        dualVal = x.dual / ((np.cos(x.real))**2)
        return DualNumber(realVal, dualVal)
    else:
        return np.tan(x)

# should this use self or take an input? Will update docstring later
def log(self, b=math.e):
    '''Calculate the natural log of a DualNumber.'''
    if isinstance(self, DualNumber):
        res = DualNumber(math.log(self.real, b), self.dual/(self.real * math.log(b)))
        return res
    else:
        return math.log(self, b)

def sqrt(self):
    '''Calculate the square root of a DualNumber.'''
    if isinstance(self, DualNumber):
        if self.real < 0:
            raise ValueError('Square root of a negative number is undefined.')
        return DualNumber(np.sqrt(self.real), (0.5 / np.sqrt(self.real))* self.dual)
    else:
        if self < 0:
            raise ValueError('Square root of a negative number is undefined.')
        return np.sqrt(self)

def asin(x):
    '''Calculate the inverse sin of a DualNumber.'''
    if isinstance(x,DualNumber):
        if x.real > 1 or x.real < -1:
            raise ValueError('Inverse sin of a number outside the range [-1,1] is undefined.')
        realVal = np.arcsin(x.real)
        dualVal = x.dual / (math.sqrt(1 - x.real**2))
        return DualNumber(realVal, dualVal)
    else:
        if x > 1 or x < -1:
            raise ValueError('Inverse sin of a number outside the range [-1,1] is undefined.')
        return np.arcsin(x)

def acos(x):
    '''Calculate the inverse cos of a DualNumber.'''
    if isinstance(x,DualNumber):
        if x.real > 1 or x.real < -1:
            raise ValueError('Inverse cos of a number outside the range [-1,1] is undefined.')
        realVal = np.arccos(x.real)
        dualVal = -x.dual / (math.sqrt(1 - x.real**2))
        return DualNumber(realVal, dualVal)
    else:
        if x > 1 or x < -1:
            raise ValueError('Inverse cos of a number outside the range [-1,1] is undefined.')
        return np.arccos(x)

def atan(x):
    '''Calculate the inverse tan of a DualNumber.'''
    if isinstance(x,DualNumber):
        realVal = np.arctan(x.real)
        dualVal = x.dual / (1 + x.real**2)
        return DualNumber(realVal, dualVal)
    else:
        return np.arctan(x)

def sinh(x):
    '''Calculate the hyperbolic sin of a DualNumber.'''
    if isinstance(x,DualNumber):
        realVal = np.sinh(x.real)
        dualVal = x.dual * np.cosh(x.real)
        return DualNumber(realVal, dualVal)
    else:
        return np.sinh(x)

def cosh(x):
    '''Calculate the hyperbolic cos of a DualNumber.'''
    if isinstance(x,DualNumber):
        realVal = np.cosh(x.real)
        dualVal = x.dual * np.sinh(x.real)
        return DualNumber(realVal, dualVal)
    else:
        return np.cosh(x)

def tanh(x):
    '''Calculate the hyperbolic tan of a DualNumber.'''
    return sinh(x) / cosh(x)

def pow(b, x):
    '''Calculate b raised to a DualNumber power.'''
    return b**x

def logistic(x, L=1.0, k= 1.0, x0=0.0):
    '''Calculate logistic function of a DualNumber.'''
    power = -k*(x - x0)
    return L / (1 + exp(power))