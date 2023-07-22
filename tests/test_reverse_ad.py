from team48_autodiff_package.autoDiff.reverse.reverse import *
from team48_autodiff_package.autoDiff.reverse.reverse_mode_helpers import *
import pytest
import numpy as np

# x1 = Variable(2, 'x1')
# y = Variable(3, 'y')
# z = Variable(4, 'z')
@pytest.mark.parametrize("test_function,test_number", [
    (lambda x: x + 5, 10),
    (lambda x: x + x + 5, 20),
    (lambda x: x + x + 5, 30),
    (lambda x: x * x + 5, 40),
    (lambda x: x * 5, 50),
    (lambda x: x ** 5, 60),
    (lambda x: x + x ** 5, 70),
    (lambda x: x + x ** 5, 80),
    (lambda x: 2**x + x ** 2, 90),
    (lambda x: 2**x + x ** 2, 0.10),
    (lambda x: x**(x - 5), 110),
    (lambda x: x**(5 - x), 120),
    (lambda x: 1/x, 130),
    (lambda x: 1/(x**2), 140),
    (lambda x: (x**2)/3, 150),
    (lambda x: (x**2)/(x + 1), 160),
    (lambda x: x/(x**2), 170),
    (lambda x: sin(x), 180),
    (lambda x: cos(x), 190),
    (lambda x: tan(x), 200),
    (lambda x: log(x), 21),
    (lambda x: x**2 + sin(x), 220),
    (lambda x: cos(x)**2 + sin(x), 230), 
    (lambda x: log(x, 10)**2, 20), 
    (lambda x: cos(log(x, 4)), 20.5), 
    (lambda x: pow(3,x), 50), 
    (lambda x: pow(5, x), 100), 
])
def test_single_variable_no_string(test_function, test_number):
    assert (np.allclose(gradient(test_function(Variable(test_number,'x')))['x'], approx(test_function, test_number)))

# need to find an effecient way to evaluate expressions with a string instead
def helper_multi_var(eq: str):
    def wrap(x,y,z):
        x,y,z = Variable(x,'x'), Variable(y, 'y'), Variable(z, 'z')
        return eval(eq)
    return wrap

@pytest.mark.parametrize("eq_str", [
    ('x*y*z'),
    ('x*y*z + cos(x)'),
    ('x*y*z * sin(x)**cos(x) + cos(y)'),
    ('log(x*y*z) - (sin(x)**cos(x))/(cos(y))'),
    ('(log(x*y*z) - (sin(x)**cos(x))/(cos(y)))**(10 * sin(z))'),
    ('z**(log(x*y*z) - (sin(x)**cos(x))/(cos(y)))**(10 * sin(z))')
])
def multi_var(eq_str):
    inputs,diff_inputs,approx1,grad,fake   = [3,4,5],[1,1,1],approx(helper_multi_var(eq_str), inputs), gradient(helper_multi_var(eq = eq_str)(*inputs)), gradient(helper_multi_var(eq = eq_str)(*diff_inputs))
    assert(all(np.allclose(approx1[i], grad[x]) for i,x in enumerate(('x','y','z'))))
    assert(not all(np.allclose(approx1[i], fake[x]) for i,x in enumerate(('x','y','z'))))

@pytest.mark.parametrize("eq_str,size", [
    ('x*y*z',3),
    ('x*y*z + cos(x)',4),
    ('x*y*z * sin(x)**cos(x) + cos(y)',5),
    ('log(x*y*z) - (sin(x)**cos(x))/(cos(y))',7),
    ('(log(x*y*z) - (sin(x)**cos(x))/(cos(y)))**(10 * sin(z))',8),
    ('z**(log(x*y*z) - (sin(x)**cos(x))/(cos(y)))**(10 * sin(z))',9)
])
def simple_vectors(eq_str,size):
    diff_inputs, inputs, approx1 =[np.random.rand(3,) for _ in range(3)], [np.random.rand(3,) for _ in range(3)], approx(helper_multi_var(eq_str), inputs)
    grad,fake = gradient(helper_multi_var(eq = eq_str)(*inputs)),gradient(helper_multi_var(eq = eq_str)(*diff_inputs))
    assert(all(np.allclose(approx1[i], grad[x]) for i,x in enumerate(('x','y','z'))))
    assert(not all(np.allclose(approx1[i], fake[x]) for i,x in enumerate(('x','y','z'))))


    