# choosing between static and dynamic graph
# I don't think we have time to do the optimzations of the static graph and since we
# are trying to emulate pytorch I think it makes sense to do the dynamic graph

# we have to use numpy here inorder to make the computation easier

# Because we are using numpy I had to refer to this a lot
# https://numpy.org/doc/stable/user/basics.subclassing.html

import numpy as np
from collections import defaultdict, deque 

class Node(np.ndarray):
    
    def _to_node(ordered = True):
        def wrap(f):
            def wrapper_f(self, other):
                if not isinstance(other, Node):
                    other = Constant(other)
                opvalue = getattr(np.ndarray, f.__name__)(self, other)

                return Operation(opvalue, f.__name__.strip('_r'),
                    self if ordered else other,
                    other if ordered else self
                )
            return wrapper_f
        return wrap

    @_to_node()
    def __add__(self, other): pass

    @_to_node()
    def __radd__(self, other): pass

    @_to_node()
    def __sub__(self, other): pass

    @_to_node(False)
    def __rsub__(self, other): pass

    @_to_node()
    def __mul__(self, other): pass

    @_to_node()
    def __rmul__(self, other): pass

    @_to_node()
    def __div__(self, other): pass

    @_to_node(False)
    def __rdiv__(self, other): pass

    @_to_node()
    def __truediv__(self, other): pass

    @_to_node(False)
    def __rtruediv__(self, other): pass

    @_to_node()
    def __pow__(self, other): pass

    @_to_node(False)
    def __rpow__(self, other): pass

def helper(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
    return np.ndarray.__new__(
            subtype, shape, dtype,
            buffer, offset, strides, order)

class Constant(Node):
    count = 0

    def __new__(subtype, val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)

        obj = helper(subtype, shape = val.shape, dtype= val.dtype, strides=val.strides, buffer=val)

        if name is not None:
            obj.name = name
        else:
            obj.name = f"const_{Constant.count}" 
            Constant.count += 1

        return obj

class Variable(Node):
    count = 0

    def __new__(subtype, val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)

        obj = helper(subtype, shape = val.shape, dtype= val.dtype, strides=val.strides, buffer=val)

        if name is not None:
            obj.name = name
        else:
            obj.name = f"var_{Variable.count}" 
            Variable.count += 1

        return obj


class Operation(Node):
    nodes_counter = defaultdict()

    def __new__(subtype, op_result, op_name, a, b=None, name=None):
        obj = helper(subtype, shape = op_result.shape, dtype= op_result.dtype, strides= op_result.strides, buffer=np.copy(op_result))
        
        obj.op_name = op_name
        obj.a = a 
        obj.b = b 

        if name is not None:
            obj.name = name
        else:
            node_number = Operation.nodes_counter.get(op_name,0)
            obj.name = f"{op_name}_{node_number}"
            Operation.nodes_counter[op_name] = node_number + 1
        
        return obj
    
class NodeQueue:
    def __init__(self, nodes):
        self.nodes = deque([node for node in nodes])
        self.nodes_ids = deque([node.name for node in nodes])

    def push(self, node):
        self.nodes.append(node)
        self.nodes_ids.append(node.name)

    def pop(self):
        node = self.nodes.popleft()
        self.nodes_ids.popleft()
        return node

    def __contains__(self, node):
        return node.name in self.nodes_ids

    def __len__(self):
        return len(self.nodes)