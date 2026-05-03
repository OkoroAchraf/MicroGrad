import math
class Value:
    def __init__(self, data, prev = set(), label='',op=''):
        self.data = data
        self.grad = 0.0
        self.prev = set(prev)
        self._backward = lambda : None
        self.op = op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), op='+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other)**-1
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + -(other)
        
    def exp(self):
        out = Value(math.exp(self.data), (self, ), op='exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**n, (self,), op='pow')

        def _backward():
            self.grad += (n*(self.data**(n-1))) * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self + -(other)
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def reLu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), op='reLu')

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if not v in visited:
                visited.add(v)
                for prev in v.prev:
                    build_topo(prev)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data})"