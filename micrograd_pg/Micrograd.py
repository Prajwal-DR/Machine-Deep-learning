import math
from graphviz import Digraph


class Value:
    """
    Value class is a object class for scalar values.

    Attributes:
    =========================

    data: float
        Scalar value,
    _prev: set()
        Contains the parent node information of any forward operations
    _op: string
        Operator type
    label: string
        operators label
    _grad: float
        gradient of the forward pass.

    Methods:
    =============================

    __repr__ : (self)
        Data representation of the Value obj
    __add__: (self,other)
        add operator function
    __radd_: (self, other)
        Order reversal function
    __mul__: (self, other)
        multiply operator function
    __rmul__: (self, other)
        Order reversal function
    __sub__: (self, other)
        Subtract operator function
    __neg__: (self)
        negative variant of self obj.
    __truediv__: (self,other)
        division operator function.
    __pow__: (self,other)
        calculate the power of value obj.
    __exp__: (self)
        exponential representation
    __tanh__: (self)
        activation function
    backward: (self):
        List of all the nodes of the neurons

    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value (data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = isinstance(other, Value) and other or Value(other)
        return self * other**-1

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2.0 * x) - 1.0) / (math.exp(2.0 * x) + 1.0)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += 1 - (t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for l in reversed(topo):
            l._backward()


class draw_nn:

    def trace(root):
        # builds a set of all nodes and edges in a graph
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    def draw_dot(root):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

        nodes, edges = draw_nn.trace(root)
        for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                # if this value is a result of some operation, create an op node for it
                dot.node(name=uid + n._op, label=n._op)
                # and connect this node to it
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            # connect n1 to the op node of n2
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
