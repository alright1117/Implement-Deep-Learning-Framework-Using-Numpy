import numpy as np

class Linear():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        self.cache = None
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}
        
    def __call__(self, X):
        return self._forward(X)
    
    def _forward(self, X):
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(np.reshape(X, (X.shape[0], -1)).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        return dX

    def _update_params(self, lr=0.001):
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']
        
class ReLU():
    def __init__(self):
        self.mask = None
    
    def __call__(self, X):
        return self._forward(X)

    def _forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def _backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

'''      
class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None
    
    def __call__(self, X):
        return self._forward(X)
    
    def _forward(self, X):
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX  
'''   

class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None
    
    def __call__(self, X):
        return self._forward(X)
    
    def _forward(self, X):
        X = np.clip( X, -50, 50 )
        X = 1 / (1 + np.exp(-X))
        self.cache = X
        return X

    def _backward(self, dout):
        X = self.cache
        dX = dout*X*(1-X)
        return dX


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


class Conv():
    def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0}
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.stride = stride
        self.pad = padding
        
        self.x = None   
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None
        
    def __call__(self, X):
        return self._forward(X)
        
    def _forward(self, x):
        FN, C, FH, FW = self.W['val'].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W['val'].reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b['val']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def _backward(self, dout):
        FN, C, FH, FW = self.W['val'].shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.b['grad'] = np.sum(dout, axis=0)
        self.W['grad'] = np.dot(self.col.T, dout)
        self.W['grad'] = self.W['grad'].transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
        
     
class MaxPool():
    def __init__(self, f, stride=2, pad=0):
        self.pool_h = f
        self.pool_w = f
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
        
    def __call__(self, X):
        return self._forward(X)
        
    def _forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def _backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx    
    
class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        pass
    def __call__(self, X):
        return self._forward(X)
    
    def _forward(self, X):
        X = X - np.max(X, axis=-1, keepdims=True)   # ????????µ¦
        return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)
  


class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax(Y_pred)
        loss = -np.sum(Y_true * np.log(prob + 1e-8)) / N
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout