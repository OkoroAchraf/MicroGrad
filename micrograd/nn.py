import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
        
        def parameters():
            return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x):
        out = sum((w* xi for w, xi in zip(self.w, x)), self.b)
        return out.reLu()

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def parameters(self):
        out = [p for neuron in self.neurons for p in neuron.parameters()]
        return out
    
    def __call__(self, x):
        neurons = [n(x) for n in self.neurons]
        return neurons

class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = [Layer(nin, nout) for nout in nouts]

    def parameters(self):
        out = [p for layer in self.layers for p in layer.parameters()]
        return out

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x