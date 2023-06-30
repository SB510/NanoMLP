from NN import *

#Define Multilayer Perceptron
n = MLP(3, [4, 4, 1])

#input training data
xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        ]
#desired output
ys = [1.0, -1.0, -1.0, 1.0]

#create NeuralNetwork wrapper
network =  NN(n, xs, ys)
network.train(200, 0.01) #train MLP 'n'

print(network.predictions)
print(network.predict([2.0, -0.5,0, 1])) #see prediction for new input