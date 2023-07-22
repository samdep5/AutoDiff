from team48_autodiff_package.dualNumber.dual_numbers import *
from team48_autodiff_package.autoDiff.forward.single_var_forward import *
from team48_autodiff_package.autoDiff.forward.AutoDiff import *
import pytest
import math

# Since it's hard to identify lambda functions by trace used increasing numbers
@pytest.mark.parametrize("test_function,test_number", [
    (lambda x: x + 5, 1),
    (lambda x: x + x + 5, 2),
    (lambda x: x + x + 5, 3),
    (lambda x: x * x + 5, 4),
    (lambda x: x * 5, 5),
    (lambda x: x ** 5, 6),
    (lambda x: x + x ** 5, 7),
    (lambda x: x + x ** 5, 8),
    (lambda x: 2**x + x ** 2, 9),
    (lambda x: 2**x + x ** 2, .10),
    (lambda x: x**(x - 5), 11),
    (lambda x: x**(5 - x), 12),
    (lambda x: 1/x, 13),
    (lambda x: 1/(x**2), 14),
    (lambda x: (x**2)/3, 15),
    (lambda x: (x**2)/(x + 1), 16),
    (lambda x: x/(x**2), 17),
    (lambda x: sin(x), 18),
    (lambda x: cos(x), 19),
    (lambda x: tan(x), 20),
    (lambda x: log(x), 21),
    (lambda x: x**2 + sin(x), 22),
    (lambda x: cos(x)**2 + sin(x), 23), 
    (lambda x: log(x, 10)**2, 2), 
    (lambda x: cos(log(x, 4)), 2.5), 
    (lambda x: sinh(log(x, 4)), .5),
    (lambda x: cosh(log(x, 4)), 3.45),
    (lambda x: tanh(log(x, 4)), 7.67),
    (lambda x: asin(x**2), .5),
    (lambda x: acos(x/ 4), 3.45),
    (lambda x: atan(x**2), 7.67),
    (lambda x: logistic(x), 3),
    (lambda x: pow(3,x), 5), 
    (lambda x: pow(5, x), 10), 
    (lambda x: logistic(sin(exp(x))), 3.5)
])
def test_single_variable_no_string(test_function, test_number):
    assert (check_error(single_derivative(test_function, test_number), single_derivative_approximation(test_function, test_number)))

@pytest.mark.parametrize("test_function,test_number", [
    (lambda x: x + 5, 1),
    (lambda x: x + x + 5, 2),
    (lambda x: x + x + 5, 3),
    (lambda x: x * x + 5, 4),
    (lambda x: x * 5, 5),
    (lambda x: x ** 5, 6),
    (lambda x: x + x ** 5, 7),
    (lambda x: x + x ** 5, 8),
    (lambda x: 2**x + x ** 2, 9),
    (lambda x: 2**x + x ** 2, .10),
    (lambda x: x**(x - 5), 11),
    (lambda x: x**(5 - x), 12),
    (lambda x: 1/x, 13),
    (lambda x: 1/(x**2), 14),
    (lambda x: (x**2)/3, 15),
    (lambda x: (x**2)/(x + 1), 16),
    (lambda x: x/(x**2), 17),
    (lambda x: sin(x), 18),
    (lambda x: cos(x), 19),
    (lambda x: tan(x), 20),
    (lambda x: log(x), 21),
    (lambda x: x**2 + sin(x), 22),
    (lambda x: cos(x)**2 + sin(x), 23), 
    (lambda x: exp(x), 24), 
    (lambda x: sin(cos(x)), 25), 
    (lambda x: exp(x**2), 26), 

])
def test_AutoDiff_class(test_function, test_number):
    f = AutoDiff(test_function)
    assert (check_error(f(test_number), single_derivative_approximation(test_function, test_number)))

def test_invalid():
    with pytest.raises(ZeroDivisionError):
        single_derivative(lambda x: 1 / x, 0)
        single_derivative(lambda x: tan(x), math.pi/2)
    with pytest.raises(ValueError):
        single_derivative(lambda x: sqrt(x), -3)
        single_derivative(lambda x: sqrt(x), -5)


@pytest.mark.parametrize("test_function,test_vec,p", [
    (lambda x,y: x + y, [5,7], [0,1]),
    (lambda x,y: x + y, [5,7], [1,1]),
    (lambda x,y: x + y, [5,7], [0,0]),
    (lambda x,y: y * x, [3,9], [0,1]),
    (lambda x,y,z: y + x - z, [3,9,5], [1,0,0]),
    (lambda x,y,z: y + x - z, [3,9,5], [1,1,1]),
    (lambda x,y,z: y + x - z, [3,9,5], [0,1,1]),
    (lambda x,y: y / x + 5, [7,7], [1,0]),
    (lambda x,y: x ** y, [5,9], [0,1]),
    (lambda x,y,z: sin(x) ** (z + cos(y)), [2,4,2], [0,0,1]),
    (lambda x,y: tan(x * y), [1,9], [1,0]),
    (lambda x,y: log(x + y), [7,70], [0,1]),
    (lambda x,y,z: exp(x) + exp(y) + tan(z), [5,9,2], [0,1,0])
])
def test_multi_vars(test_function, test_vec, p):
    f = AutoDiff(test_function)
    assert (check_error(f(test_vec, p), derivative_approximation(test_function, test_vec, p)))

# making list of lambda functions doesn't work so declared here
a1 = lambda x,y: x + y
b2 = lambda x,y: y * x
c3 = lambda x,y,z: y + x - z
d4 = lambda x,y: y / x + 5
e5 = lambda x,y: x ** y
f6 = lambda x,y,z: sin(x) ** (z + cos(y))
g7 = lambda x,y: tan(x * y)
h8 = lambda x,y: log(x + y)
i9 = lambda x,y,z: exp(x) + exp(y) + tan(z)

@pytest.mark.parametrize("test_function,test_vec,p", [
    ([c3, f6], [2,2,2], [0,1,0]),
    ([a1,b2,d4], [5,7], [1,1]),
    ([g7,h8], [5,7], [0,0]),
    ([b2,h8,g7], [3,9], [0,1]),
    ([a1,b2,e5,g7,h8], [3,9], [1,0]),
    ([i9,c3,f6], [2,2,2], [1,1,1]),
])
def test_multi_func(test_function, test_vec, p):
    f = AutoDiff(test_function)
    assert (np.allclose(f(test_vec, p), derivative_approximation(test_function, test_vec, p)))

@pytest.mark.parametrize("test_function,test_vec,val", [
    (lambda x,y: x + y, [7,2], 9),
    (lambda x,y,z: x + y + tan(z), [5,9,2], 11.8149601)
])
def test_multi_vars(test_function, test_vec, val):
    f = AutoDiff(test_function)
    assert (np.allclose(f.evaluate(test_vec), val))

@pytest.mark.parametrize("test_function,test_val,val", [
    ('x + 5', 20, 1),
    ('x**2 + 5', 2, 4),
    ('x**3 + 5', 2, 12),
    ('x + x**3 + 5', 2, 13)
])
def test_str1(test_function, test_val, val):
    f = AutoDiff(test_function)
    assert f(test_val) == val


@pytest.mark.parametrize("test_function,test_val,val,p", [
    ('x**2 + y', [10,10], 20, 0)
])
def test_str(test_function, test_val, val,p):
    f = AutoDiff(test_function)
    assert f(test_val,0) == val