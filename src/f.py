from team48_autodiff_package.autoDiff.forward.AutoDiff import *
from team48_autodiff_package.autoDiff.forward.single_var_forward import *
from team48_autodiff_package.autoDiff.forward.single_var_forward import *

d3 = 2
d4 = 4

def func1(x, y):
    return x + y

def func4(x):
    return x ** 2
def func2(x):
    return sin(x)

def func3(x, y):
    return x * y + 2

# Single function single value (vector or scalar accepted)
forward_diff_outs = AutoDiff(func2)
print(forward_diff_outs(d3, 1))
print(forward_diff_outs.gradient(d4))
print(forward_diff_outs.evaluate(d3))

#Multiple functions
forward_diff_outs = AutoDiff([func4, func2])
xValues = d3
print(forward_diff_outs(xValues, 1))
print(forward_diff_outs.gradient(xValues))
print(forward_diff_outs.evaluate(xValues))

# Multiple variables single function
forward_diff_outs = AutoDiff(func2)
xValues = np.array([d3, d4])
print(forward_diff_outs(xValues))
print(forward_diff_outs.gradient(xValues))
print(forward_diff_outs.evaluate(xValues))

# Multiple variables multiple functions
forward_diff_outs = AutoDiff([func3, func2])
xValues = [d3, d3]
print(forward_diff_outs(xValues, [1, 0]))
print(forward_diff_outs.gradient(xValues))
print(forward_diff_outs.evaluate(xValues))
