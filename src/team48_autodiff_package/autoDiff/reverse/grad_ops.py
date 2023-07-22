from team48_autodiff_package.nodeGraph.node_operations import *

def add_grad(prev, node):
   '''Addition for nodes'''
   return (prev, prev)
 
def sub_grad(prev, node):
   '''subtraction for nodes'''
   return (prev, -1 * prev)
 
def mul_grad(prev, node):
   '''multiplication for nodes'''
   return (prev * node.b,prev * node.a)
 
def truediv_grad(prev, node):
   '''division for nodes'''
   return (
       prev / node.b,
       -1 * prev * node.a / node.b ** 2
   )
 
def pow_grad(prev, node):
   '''power operation for nodes'''
   return (
       prev * node.b * (node.a ** (node.b - 1)),
       prev * node * log(node.a)
   )
 
def exp_grad(prev, node):
   '''e ^ node'''
   return (prev * node, None)
 
def log_grad(prev, node):
   '''log operation for nodes'''
   return (prev * (1. / node.a), None)
 
def sin_grad(prev, node):
   '''sine for nodes'''
   return (prev * cos(node.a), None)
 
def cos_grad(prev, node):
   '''cosine for nodes'''
   return (-1 * prev * sin(node.a), None)
 
def tan_grad(prev, node):
   '''tangent operation for nodes'''
   return (prev * 1/(cos(node.a)**2), None)

# These were meant for back prop, but we were never able to get to this part
def sum_grad(prev, node):
   '''summation for nodes'''
   return (prev * np.ones_like(node.a), None)
 
def dot_grad(prev, node):
   '''dot product for nodes'''
   prev_adj = prev
   op_a = node.a
   op_b = node.b
   return (
       np.dot(prev_adj, op_b.T),
       np.dot(op_a.T, prev_adj)
   )
