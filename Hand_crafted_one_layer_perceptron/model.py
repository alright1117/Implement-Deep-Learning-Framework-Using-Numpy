import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class perceptron():
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.train_acc_top1 = []
        self.train_acc_top5 = []
        self.val_acc_top1 = []
        self.val_acc_top5 = []
        self.losses = []
        self.initialize_parameters()

 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
 
    def initialize_parameters(self):
        np.random.seed(1)
        self.parameters["W"] = np.random.randn(self.layers_size[1], self.layers_size[0]) / np.sqrt(
                self.layers_size[0])
        self.parameters["b"] = np.zeros((self.layers_size[1], 1))
 
    def forward(self, X):
        store = {}
 
        Z = self.parameters["W"].dot(X.T) + self.parameters["b"]
        A = self.softmax(Z)
 
        return A
 
 
    def backward(self, X, Y, A):
 
        derivatives = {}
    
        dZ = A - Y.T
        dW = dZ.dot(X) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        
        derivatives["dW"] = dW
        derivatives["db"] = db
 
        return derivatives
 
    def train(self, X, Y, learning_rate=0.05):
        np.random.seed(1)
 
        self.n = X.shape[0]

        A = self.forward(X)
        loss = -np.mean(Y * np.log(A.T+ 1e-8))
        derivatives = self.backward(X, Y, A)
        self.parameters["W"] = self.parameters["W"] - learning_rate * derivatives["dW"]
        self.parameters["b"] = self.parameters["b"] - learning_rate * derivatives["db"]
 
        return loss
 
    def predict(self, X, Y, k):
        y_hat = self.forward(X)
        y_hat_k = y_hat.argsort(axis=0).T[:,-k:]
        Y = np.argmax(Y, axis=1).reshape(((len(Y),1)))
        top_k_accuracy = (y_hat_k == Y).sum() / len(Y)
        return top_k_accuracy * 100
 
    def plot_loss(self):
        fig = plt.figure()
        plt.plot(list(range(0, len(self.losses) * 50, 50)) , self.losses)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        fig.savefig("result/loss.png")
        
    def plot_acc(self):
        sns.set_style("darkgrid")    

        fig = plt.figure()
        x = list(range(0, len(self.train_acc_top1) * 50, 50))
        ax = sns.lineplot(x = x, y=self.train_acc_top1, color='seagreen', label = 'Train Top 1 accuracy')
        sns.lineplot(x = x, y=self.train_acc_top5, color='royalblue', label = 'Train Top 5 accuracy')
        sns.lineplot(x = x, y=self.val_acc_top1, color='red', label = 'Validation Top 1 accuracy')
        sns.lineplot(x = x, y=self.val_acc_top5, color='gold', label = 'Validation Top 5 accuracy')
        ax.set_xlabel("Epoch", size = 14)
        ax.set_ylabel("Accuracy", size = 14)
        ax.set_title("Train vs Validation", size = 14, fontweight='bold')
        ax.legend()
        fig.set_figheight(6)
        fig.set_figwidth(16)
        fig.savefig("result/acc.png")
        
    def load(self, path):
        self.parameters = np.load(path, allow_pickle=True).item()
        print('Load successfully!')
        
    def save(self, path):
        np.save(path, self.parameters)
        print('Save successfully!')