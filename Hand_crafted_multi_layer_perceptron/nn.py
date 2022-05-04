import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        self.parameters = {'W': np.random.randn(input_size, output_size) / np.sqrt(input_size),
                           'b': np.zeros((output_size))}
        
    def __call__(self, Z):
        return self.forward(Z)
    
    def forward(self, X):
        self.X = X
        self.n = X.shape[0]
        return X.dot(self.parameters['W']) + self.parameters['b']
    
    def backward(self, dZ):
        dA = dZ.dot(self.parameters['W'].T)
        dW = self.X.T.dot(dZ) / self.n
        db = 1. / self.n * np.sum(dZ, axis=0)
        return dA, dW, db
        
class Sigmoid:
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.z = 1 / (1 + np.exp(-X))
        return self.z

    def backward(self):
        return self.z * (1 - self.z)
    
class CrossEntropy:
    def __call__(self, Z, Y):
        return self.forward(Z, Y)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)
    
    def forward(self, Z, Y):
        self.A = self.softmax(Z)
        self.Y = Y
        L = -np.mean(Y * np.log(self.A+ 1e-8))
        return L
    
    def backward(self):
        dZ = self.A - self.Y
        return dZ
    
class Model:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.l1 = Linear(input_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.l2 = Linear(hidden_size, output_size)
        self.parameters = {'l1': self.l1.parameters, 'l2': self.l2.parameters}
        self.lr = lr
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l2(x)
        return x
    
    def backward(self, dZ):
        dA, dW2, db2 = self.l2.backward(dZ)
        dZ = dA * self.sigmoid.backward()
        dA, dW1, db1 = self.l1.backward(dZ)
        return [dW1, db1, dW2, db2]
    
    def load(self, path):
        self.parameters = np.load(path, allow_pickle=True).item()
        self.l1.parameters = self.parameters['l1']
        self.l2.parameters = self.parameters['l2']
        print('Load successfully!')
        
    def save(self, path):
        np.save(path, self.parameters)
        print('Save successfully!')
    
    def step(self, dZ):
        grad = self.backward(dZ)
        for layer in self.parameters:
            for parameter in self.parameters[layer]:
                self.parameters[layer][parameter] -= self.lr * grad.pop(0)
