from Network import *



class NN:
    def __init__(self, model, trainX, trainY):
        self.model = model
        self.trainX = trainX
        self.trainY = trainY
        self.predictions = []
        self.loss = 0
    def train(self, iterations, learn_rate):    
        for k in range(iterations):
            # forward pass
            self.predictions = [self.model(x) for x in self.trainX]
            self.loss = sum((yout - ygt)**2 for ygt, yout in zip(self.trainY, self.predictions))
            
            # backward pass
            for p in self.model.parameters():
                p.grad = 0.0
            self.loss.backward()
            
            # update
            for p in self.model.parameters():
                p.data += -learn_rate * p.grad
            
            #print(k, self.loss.data)
    def predict(self, input_datapoint):
        return self.model(input_datapoint)