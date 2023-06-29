from NN import *

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        ]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

network =  NN(n, xs, ys)
network.train(200)
print(network.predictions)