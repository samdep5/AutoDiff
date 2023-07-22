from team48_autodiff_package.autoDiff.forward.single_var_forward import *
from team48_autodiff_package.dualNumber.dual_numbers import *
import math
import pytest

def test_check_error():
    # err*(1 + abs(approx))
    assert check_error(1e-6*(2), 1) == False
    assert check_error(1.0000005, 1) == True
    assert check_error(1, 2) == False
    assert check_error(1.0, 1) == True
    assert check_error(2.0, 2.00004) == False
    
def test_derivative_approximation():
    assert check_error(single_derivative_approximation(lambda x: x + 5, 1), 1)
    assert check_error(single_derivative_approximation(lambda x: x + 5, 1), 1)
    assert check_error(single_derivative_approximation(lambda x: x + x + 5, 2), 2)
    assert check_error(single_derivative_approximation(lambda x: x + x + 5, 3), 2)
    assert check_error(single_derivative_approximation(lambda x: x * x + 5, 4), 8)
    assert check_error(single_derivative_approximation(lambda x: x * 5, 5), 5)
    assert check_error(single_derivative_approximation(lambda x: x ** 5, 6), 6480)
    assert check_error(single_derivative_approximation(lambda x: x + x ** 5, 7), 12006)
    assert check_error(single_derivative_approximation(lambda x: x + x ** 5, 8), 20481)
    assert check_error(single_derivative_approximation(lambda x: 2**x + x ** 2, 9), (2**9)*math.log(2) + 18)
    assert check_error(single_derivative_approximation(lambda x: 2**x + x**2, .10), (2**.1)*math.log(2) + .1*2)
    assert check_error(single_derivative_approximation(lambda x: x**(x - 5), 11), 1771561*(math.log(11) + 6/11))
    assert check_error(single_derivative_approximation(lambda x: x**(5 - x), 12), (-math.log(12) - 7/12) / 35831808)
    assert check_error(single_derivative_approximation(lambda x: 1/x, 13), -1/169)
    assert check_error(single_derivative_approximation(lambda x: 1/(x**2), 14), -1/1372)
    assert check_error(single_derivative_approximation(lambda x: (x**2)/3, 15), 10)
    assert check_error(single_derivative_approximation(lambda x: (x**2)/(x + 1), 16), 288/289)
    assert check_error(single_derivative_approximation(lambda x: x/(x**2), 17), -1/289)
    assert check_error(single_derivative_approximation(lambda x: sin(x), 18), math.cos(18) )
    assert check_error(single_derivative_approximation(lambda x: cos(x), 19), -math.sin(19))
    assert check_error(single_derivative_approximation(lambda x: tan(x), 20), 1/ (math.cos(20))**2)
    assert check_error(single_derivative_approximation(lambda x: log(x), 21), 1/ 21)
    assert check_error(single_derivative_approximation(lambda x: x**2 + sin(x), 22), 44 + math.cos(22))
    assert check_error(single_derivative_approximation(lambda x: cos(x)**2 + sin(x), 23), math.cos(23) - math.sin(46))