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
           
            try:
                 # forward pass
                self.predictions = [self.model(x) for x in self.trainX]
                self.loss = sum((yout - ygt)**2 for ygt, yout in zip(self.trainY, self.predictions))
                # backward pass
                for p in self.model.parameters():
                    p.grad = 0.0
                # update
                self.loss.backward()
                for p in self.model.parameters():
                    p.data += -learn_rate * p.grad
            
            except:
                print("Warning, an error has occured, it's likley the training dataset is too large, starting to segement training")
                for i in range(100,len(self.trainY), 100):
                    #print(self.trainX[i-100:i])
                    self.predictions = [self.model(x) for x in self.trainX[i-100:i]]
                    self.loss = sum((yout - ygt)**2 for ygt, yout in zip(self.trainY[i-100:i], self.predictions))
                    # backward pass
                    for p in self.model.parameters():
                        p.grad = 0.0
                    self.loss.backward()
                    for p in self.model.parameters():
                        p.data += -learn_rate * p.grad
            
            print(k, self.loss.data)
    def predict(self, input_datapoint):
        return self.model(input_datapoint)
    

