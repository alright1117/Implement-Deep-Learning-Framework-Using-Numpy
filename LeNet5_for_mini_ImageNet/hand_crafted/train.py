import numpy as np

from dataset import load_data
from nn import CrossEntropyLoss
from optimizer import SGDMomentum, SGD
from model import LeNet, LeNet_plus
from plot import plot_acc, plot_loss
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type = str, default = 'LeNet')
parser.add_argument("--iters", type = int, default = 1000)
parser.add_argument("--train_batch", type = int, default = 128)
parser.add_argument("--val_batch", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 0.001)

args = parser.parse_args()


if args.model == 'LeNet':
    model = LeNet()
    size = 128
else:
    model = LeNet_plus()
    size = 126
    
val_x, val_y = load_data('val',batch=args.val_batch, size=size)
val_y = np.argmax(val_y, axis=1).reshape(((len(val_y),1)))    
    
losses = []
train_accs = []
val_accs = []
iters = []
best_acc = 0.0
#optim = SGD(model.get_params(), lr=0.0001, reg=0)
optim = SGDMomentum(model.get_params(), lr=args.lr, momentum=0.80, reg=0.00003)
criterion = CrossEntropyLoss()

# TRAIN

for i in range(args.iters):

    X_batch, Y_batch = load_data('train',batch=args.train_batch, size=size)
    # forward, loss, backward, step
    Y_pred = model(X_batch)
    
    loss, dout = criterion.get(Y_pred, Y_batch)
    a = model.backward(dout)
    optim.step()
    
    Y_pred = Y_pred.argsort(axis=1)[:,-1:]
    Y_batch = np.argmax(Y_batch, axis=1).reshape(((len(Y_batch),1)))
    train_acc = (Y_pred == Y_batch).mean()
    print("iter: %s, loss: %s, acc: %s" % (i, loss, train_acc))
    losses.append(loss)

    if (i % 10) == 0:
        val_y_pred = model(val_x)
        val_y_pred = val_y_pred.argsort(axis=1)[:,-1:]
        val_acc = (val_y_pred == val_y).mean()
        print("iter: %s, loss: %s, train_acc: %s, val_acc: %s" % (i, loss, train_acc, val_acc))
        if val_acc > best_acc:
            model.save(args.model + '.npy')
            best_acc = val_acc
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        iters.append(i)
        

plot_acc(train_accs, val_accs, iters, args.model + '_acc.png')
plot_loss(losses, args.model + '_loss.png')