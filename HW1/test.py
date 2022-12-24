from ast import main
import numpy as np


class Perceptron():
    def __init__(self, learning_rate = 0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.actication_func = self.forward
        self.weights = None
        self.bias = None
  

    def forward(self, x):  
        return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #initalize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range (self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_funct(linear_output)
               
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update 

                

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        prediction = self.forward(linear)
        return prediction 

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
        
    




if __name__ == '__main__':
    layer_1 = Perceptron(layer_1, [.15,.20,.35])
    X = [.05, .1, layer_1.bias]
    print("Here is the h1: ")
    #This should output 
    h1 = Perceptron.forward(layer_1, X)
    
    print(h1)
 
