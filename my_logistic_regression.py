import numpy as np
from scipy.spatial.distance import cdist

class MyLogisticRegression():
    
    def __init__(self, dim, lr, training_iter):
        self.param = np.zeros(dim)
        self.lr = lr
        self.training_iter = training_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_grad(self, x, y):
        sig_val = self.sigmoid(x.dot(self.param))
        #err = sig_val - y 
        # dot product of x (sigmoid(cita x) - y)
        #grad = (x * np.expand_dims(err, axis=-1)) / 
        # aggreagte the gradients
        #grad = np.sum(grad, axis=0)
        gradient = x.T.dot(sig_val - y) / len(y)
        
        return gradient

    def predict(self, test_x):
        sig_val = self.sigmoid(test_x.dot(self.param))
        predictions = (sig_val >= 0.5).astype(int)
        # make sure it return a numpy
        return predictions
    
    def calculate_the_error(self, x, y):
        predictions = self.predict(x)
        # calculate the acc, precision and recall
        acc = np.sum(predictions == y) / len(y)
        return 1-acc

        # m = len(y)
        # h = self.sigmoid(x.dot(self.param))
        # predictions = (h >= 0.5).astype(int)
        # incorrect = np.sum(predictions != y)
        # error = incorrect / m
        # return error
    
    def train(self, train_x, train_y, test_x, test_y):
        for i in range(self.training_iter):
            # calculate the gradient
            gradients = self.calculate_grad(train_x, train_y)
            # update the model
            self.param -= self.lr * gradients
            
            # see the train error and test error per 10 iteration
            if i % 1000 ==0:
                train_err = self.calculate_the_error(train_x, train_y)
                test_err = self.calculate_the_error(test_x, test_y)
                print("train_err:", train_err, " test_err:", test_err)
        
