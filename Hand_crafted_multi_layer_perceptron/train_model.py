import sys
import numpy as np
from nn import *
from plot import *
dataset_dir = '/home/ncku01/DL/hw1/images_feature/'
lr = 0.1
input_size = 3780
hidden_size = 1000
output_size = 50
iters = 5000
for args in sys.argv:
    # --- data path ---
    if args.startswith("dataset_dir"):
        dataset_dir = args.split("=")[1]

    if args.startswith("lr"):
        lr = float(args.split("=")[1])

    if args.startswith("layer"):
        layer = args.split("=")[1:]
    
    if args.startswith("iters"):
        iters = int(args.split("=")[1])
        
train_x, train_y = np.load(dataset_dir + 'train_x.npy'), np.load(dataset_dir + 'train_y.npy')
val_x, val_y = np.load(dataset_dir + 'val_x.npy'), np.load(dataset_dir + 'val_y.npy')

model = Model(input_size, hidden_size, output_size, lr)
model.load('model.npy')
loss = CrossEntropy()

losses = []
train_top1_acc = []
train_top5_acc = []
val_top1_acc = []
val_top5_acc = []
best_acc = 0

for i in range(iters):
    
    pred_y = model(train_x)
    train_loss = loss(pred_y, train_y)
    model.step(loss.backward())

    if i % 10 == 0:
        
        losses.append(train_loss)
    
        # train Accuracy
        train_acc = cal_acc(pred_y, train_y)
        train_top1_acc.append(train_acc[0])
        train_top5_acc.append(train_acc[1])
        
        # Val Accuracy
        pred_y_val = model(val_x)
        val_acc = cal_acc(pred_y_val, val_y)
        val_top1_acc.append(val_acc[0])
        val_top5_acc.append(val_acc[1])
        
        print('Iter %d: loss %f acc: %f' % (i, train_loss, val_acc[1]))
        
        if val_acc[1] > best_acc:
            best_acc = val_acc[1]
            model.save('model.npy')
            
    if (i % 1000 == 0) and i != 0:
        plot_acc(train_acc, val_acc, 10)
        plot_loss(losses, 10)
        
val_acc = [val_top1_acc, val_top5_acc]
train_acc = [train_top1_acc, train_top5_acc]
plot_acc(train_acc, val_acc, 10)
plot_loss(losses, 10)