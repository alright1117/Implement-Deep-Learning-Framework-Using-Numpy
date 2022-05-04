from abc import ABCMeta, abstractmethod
from nn import *

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, X):
        pass
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass
    
    
class LeNet(Net):

    def __init__(self):
        self.conv1 = Conv(3, 6, 5)
        self.sig1 = Sigmoid()
        self.pool1 = MaxPool(2,2)
        self.conv2 = Conv(6, 16, 5)
        self.sig2 = Sigmoid()
        self.pool2 = MaxPool(2,2)
        #self.FC1 = Linear(13*13*16, 120) # size 64
        self.FC1 = Linear(29*29*16, 120) # size 128
        self.FC2 = Linear(120, 84)
        self.FC3 = Linear(84, 50)

        self.p2_shape = None
        
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        h1 = self.conv1(X)
        a1 = self.sig1(h1)
        p1 = self.pool1(a1)
        h2 = self.conv2(p1)
        a2 = self.sig2(h2)
        p2 = self.pool2(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten
        h3 = self.FC1(fl)
        h4 = self.FC2(h3)
        h5 = self.FC3(h4)
        return h5

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2._backward(dout)
        dout = self.sig2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.sig1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params
        
    def save(self, path):
        weights = self.get_params()
        np.save(path, weights)
        print('Save successfully!')
        
    def load(self, path):
        weights = np.load(path, allow_pickle=True)
        self.set_params(weights)
        print('Load successfully!')
        
class LeNet_plus(Net):

    def __init__(self):
        self.conv1 = Conv(3, 6, 3)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2)
        self.conv2 = Conv(6, 16, 3)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2,2)
        self.conv3 = Conv(16, 20, 3)
        self.ReLU3 = ReLU()
        #self.FC1 = Linear(6*6*20, 120) # size 62
        self.FC1 = Linear(28*28*20, 120) # size 126
        self.ReLU4 = ReLU()
        self.FC2 = Linear(120, 84)
        self.ReLU5 = ReLU()
        self.FC3 = Linear(84, 50)

        
        
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        h1 = self.conv1(X)
        a1 = self.ReLU1(h1)
        p1 = self.pool1(a1)
        h2 = self.conv2(p1)
        a2 = self.ReLU2(h2)
        p2 = self.pool2(a2)
        h3 = self.conv3(p2)
        a3 = self.ReLU3(h3)
        self.a3_shape = a3.shape
        fl = a3.reshape(X.shape[0],-1) # Flatten
        h4 = self.FC1(fl)
        a4 = self.ReLU4(h4)
        h5 = self.FC2(a4)
        a5 = self.ReLU5(h5)
        h6 = self.FC3(a5)
        return h6

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.ReLU5._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.a3_shape) # reshape
        dout = self.ReLU3._backward(dout)
        dout = self.conv3._backward(dout)
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params
    
    def save(self, path):
        weights = self.get_params()
        np.save(path, weights)
        print('Save successfully!')
        
    def load(self, path):
        weights = np.load(path, allow_pickle=True)
        self.set_params(weights)
        print('Load successfully!')