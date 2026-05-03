# Micrograd

A tiny scalar-valued autograd engine and neural network library, written from scratch in pure Python. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The whole point of this project is **to understand how `loss.backward()` actually works** in libraries like PyTorch — by building it yourself in ~100 lines of code.

---

## What is autograd?

When you train a neural network, you compute a **loss** (a single number telling you how wrong the network is) and then ask: "for every parameter in the network, how should I nudge it to make the loss smaller?" The answer is a **gradient**: a number per parameter saying which direction reduces the loss and by how much.

Computing those gradients by hand is tedious and error-prone. **Autograd** does it for you automatically:

1. As you build up the math (`a + b`, `c * d`, `relu(x)`, ...), the engine secretly records a graph of every operation.
2. When you call `.backward()` on the final result, the engine walks that graph backwards and applies the chain rule from calculus to fill in the gradient for every value that contributed.

Micrograd does this for plain Python `float`s — one number at a time, no tensors, no GPUs. Slow, but the simplest possible thing that works.

---

## Project layout

```
Micrograd/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py    # the Value class — autograd lives here
│   ├── nn.py        # Neuron, Layer, MLP — built on top of Value
│   └── main.py      # tiny demo script
├── test_engine.py   # pytest tests, cross-checked against PyTorch
└── README.md
```

---

## Quick start

### Requirements

- Python 3.8+
- For running tests: `pytest` and `torch` (used to verify our gradients match PyTorch's)

```bash
pip install pytest torch
```

### Your first gradient

```python
from micrograd.engine import Value

# build a small expression
a = Value(2.0)
b = Value(-3.0)
c = a * b + b**2
# c.data == -3.0

# compute gradients of c with respect to every input
c.backward()

print(a.grad)   # dc/da = b   = -3.0
print(b.grad)   # dc/db = a + 2b = 2 + (-6) = -4.0
```

That's it. You wrote an expression, called `.backward()`, and now every `Value` involved knows how it influences the result.

---

## How `Value` works

Every `Value` holds four things:

| field       | what it is                                                      |
| ----------- | --------------------------------------------------------------- |
| `data`      | the actual number (e.g. `3.0`)                                  |
| `grad`      | the gradient of the final output w.r.t. this value (starts `0`) |
| `prev`      | the set of `Value`s that produced this one                      |
| `_backward` | a closure that knows how to push gradient back to its parents   |

### Forward pass: building the graph

When you write `c = a * b`, the `__mul__` method:

1. Creates a new `Value` whose `data = a.data * b.data`.
2. Records `(a, b)` as its parents in `prev`.
3. Attaches a `_backward` closure that, if asked, will distribute _its_ gradient back to `a` and `b` according to the chain rule.

For multiplication, the chain rule says: if `c = a * b`, then `dc/da = b` and `dc/db = a`. So the closure does:

```python
a.grad += b.data * c.grad
b.grad += a.data * c.grad
```

Each operator (`+`, `-`, `*`, `/`, `**`, `exp`, `reLu`) has its own version of this.

### Backward pass: walking the graph

`Value.backward()` does two things:

1. **Topological sort** of the graph, so we process every node strictly _after_ all the nodes it depends on.
2. **Seed the output gradient** with `1.0` (the derivative of any value with respect to itself), then walk the sorted list in reverse, calling each node's `_backward` closure.

By the time we reach the inputs, every `Value.grad` holds `d(output)/d(this_value)`.

---

## Supported operations

```python
a + b      a - b      a * b      a / b
a ** n     -a         a.reLu()   a.exp()
```

All of these work with mixed `Value` / `float` operands (`2 * a`, `a + 3`, etc.) thanks to the `__radd__` / `__rmul__` / `__rsub__` / `__rtruediv__` reverse-operator hooks.

> Note: the ReLU activation is spelled `reLu()` in this codebase (capital `L`).

---

## Building a tiny neural network

`micrograd/nn.py` builds three small layers of abstraction on top of `Value`:

- `Neuron(nin)` — `nin` weights, one bias, ReLU activation.
- `Layer(nin, nout)` — a list of `nout` neurons, each receiving `nin` inputs.
- `MLP(nin, nouts)` — a multi-layer perceptron. `nouts` is the list of layer sizes.

```python
from micrograd.nn import MLP

# 3 inputs → hidden layer of 4 → hidden layer of 4 → 1 output
net = MLP(3, [4, 4, 1])

x = [1.0, 2.0, 5.0]
y = net(x)              # forward pass
print(y)                # the network's prediction

# every weight and bias as a Value:
print(len(net.parameters()))
```

To **train** the network, you would:

1. Compute a loss like `loss = (y - target)**2`.
2. Zero out gradients with `net.zero_grad()` (gradients accumulate by default).
3. Call `loss.backward()`.
4. Nudge each parameter against its gradient: `p.data -= lr * p.grad`.
5. Repeat.

---

## Running the tests

The tests in `test_engine.py` build the same expression in micrograd and in PyTorch, then assert that both the forward values and the gradients match.

```bash
cd /path/to/Micrograd
pytest test_engine.py -v
```

If everything is implemented correctly you should see something like:

```
test_engine.py::test_sanity_check PASSED
test_engine.py::test_more_ops PASSED
================ 2 passed in 0.5s ================
```

> **Why run from the project root?** Python adds the script's directory to `sys.path`, not its parent. Running `python test/test_engine.py` from outside the project root means Python can't find the `micrograd` package. `pytest` (run from the root) handles this for you, or you can set `PYTHONPATH=.`.

---

## Where to go next

- Read `engine.py` line by line — it's about 100 lines and every line matters.
- Read `nn.py` and convince yourself a "neural network" is just a stack of multiplications, additions, and ReLUs.
- Watch Karpathy's lecture **"The spelled-out intro to neural networks and backpropagation: building micrograd"** on YouTube for a 2-hour walkthrough.
- Try training the MLP on a small dataset (e.g. classifying points in 2D) and watch the loss go down.

---

## Credits

This project is a from-scratch reimplementation for learning purposes, following the design of Andrej Karpathy's original [micrograd](https://github.com/karpathy/micrograd).
