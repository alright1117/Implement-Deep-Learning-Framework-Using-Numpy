import sys
import numpy as np
from model import perceptron

dataset_dir = 'images_feature/'
lr = 0.05
layer = [3780, 50]
iters = 15000
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

model = perceptron(layer)

best_acc = 0
for i in range(iters):
    
    loss = model.train(train_x, train_y, learning_rate = lr)
    
    if i % 50 == 0:
        model.losses.append(loss)
        val_acc = model.predict(val_x, val_y, 5)
        model.val_acc_top5.append(val_acc)
        model.val_acc_top1.append(model.predict(val_x, val_y, 1))
        train_acc = model.predict(train_x, train_y, 5)
        model.train_acc_top5.append(train_acc)
        model.train_acc_top1.append(model.predict(train_x, train_y, 1))
        print('Iter %d: loss %f acc: %f' % (i, loss, val_acc))
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save('model.npy')

model.plot_acc()
model.plot_loss()