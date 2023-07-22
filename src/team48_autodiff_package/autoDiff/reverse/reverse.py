from team48_autodiff_package.nodeGraph.node import *
from team48_autodiff_package.nodeGraph.node_operations import *
import team48_autodiff_package.autoDiff.reverse.grad_ops as grad_ops

def resize_for_node(node, adjoint):
    '''
    Resize (recalculate) the adjoint for a node (i.e. the sensitivity of the node)
    
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

    # This was more important to calculating gradients for vectors that get broadcast in backprop
    # We did not get to back prop here
    resize_adjoint = adjoint 
    if node.shape != adjoint.shape:
        dimensions_diff = np.abs(adjoint.ndim - node.ndim)
        # Don't take the if statements out, summing is expensive
        if dimensions_diff != 0:
            resize_adjoint = sum(adjoint, axis=tuple(range(dimensions_diff)))
            ones = tuple([axis for axis, size in enumerate(node.shape) if size == 1])
            if len(ones) != 0:
                resize_adjoint = sum(resize_adjoint,  axis=ones, keepdims=True)

    return resize_adjoint  

def gradient(root):
    '''
    Finds derivative of each variable in an equation
    
    Parameters
    ----------
    root :
        node from which backpropagation starts
    
    Returns
    -------
    dictionary with variable names and the derivative
    of the function with respect to them
    '''

    # We were to do backprop bfs here, did not get around, maybe better design for other reverse mode
    v = defaultdict()
    v[root.name] = Constant(np.ones(root.shape))
    grad = {}
    queue = NodeQueue([root])

    while len(queue) > 0:
        current_node = queue.pop()
        if isinstance(current_node, Constant):
            continue
        elif isinstance(current_node, Variable):
            grad[current_node.name] = v[current_node.name]
            continue
        elif isinstance(current_node, Operation):
            current_v = v[current_node.name]
            op = current_node.op_name
            v_1, v_2 = getattr(grad_ops, f'{op}_grad')(current_v, current_node)
            
            v[current_node.a.name] = resize_for_node(
                current_node.a,
                v.get(current_node.a.name, 0) + v_1
            )
            if current_node.a not in queue:
                queue.push(current_node.a)

            if current_node.b is not None:
                v[current_node.b.name] = resize_for_node(
                    current_node.b,
                    v.get(current_node.b.name, 0)+ v_2
                )
                if current_node.b not in queue:
                    queue.push(current_node.b)
    return grad

def multi_function_grad(fx):
    ret = []
    if isinstance(fx, list):
        for f in fx:
            ret.append(gradient(f))
    return ret

