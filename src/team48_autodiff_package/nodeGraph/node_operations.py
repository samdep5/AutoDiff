from team48_autodiff_package.nodeGraph.node import *

def fix_np(f):
    def wrap(x, name= None, *args, **kwargs):
        if not isinstance(x, Node):
            x = Constant(x)
        return Operation(getattr(np, f.__name__)(x, *args, **kwargs), f.__name__, x, name=name)
    return wrap
    
@fix_np
def sum(x,axis=None, keepdims=False, name=None): pass

@fix_np
def log(x, name=None): pass

@fix_np
def exp(x, name=None): pass

@fix_np
def sin(x, name=None): pass

@fix_np
def cos(x, name=None): pass

@fix_np
def tan(x, name=None): pass

@fix_np
def asin(x, name=None): pass

@fix_np
def acos(x, name=None): pass

@fix_np
def atan(x, name=None): pass

@fix_np
def sinh(x, name=None): pass

@fix_np
def cosh(x, name=None): pass

@fix_np
def tanh(x, name=None): pass