from nn import MLP

x = [1.0, 2.0, 5.0]
n = MLP(3, [4, 4, 1])
print(n(x))
print(n.parameters())