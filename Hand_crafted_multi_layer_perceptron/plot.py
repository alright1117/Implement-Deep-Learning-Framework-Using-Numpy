import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

def cal_acc(pred_y, y):
    pred_y_top1 = pred_y.argsort(axis=1)[:,-1:]
    pred_y_top5 = pred_y.argsort(axis=1)[:,-5:]
    y = np.argmax(y, axis=1).reshape(((len(y),1)))
    top_1 = (pred_y_top1 == y).sum() / len(y)
    top_5 = (pred_y_top5 == y).sum() / len(y)
    
    return [top_1, top_5]

def plot_acc(train_acc, val_acc, epoch):
    sns.set_style("darkgrid")    

    fig = plt.figure()
    x = list(range(0, len(train_acc[0]) * epoch, epoch))
    ax = sns.lineplot(x = x, y=train_acc[0], color='seagreen', label = 'Train Top 1 accuracy')
    sns.lineplot(x = x, y=train_acc[1], color='royalblue', label = 'Train Top 5 accuracy')
    sns.lineplot(x = x, y=val_acc[0], color='red', label = 'Validation Top 1 accuracy')
    sns.lineplot(x = x, y=val_acc[1], color='gold', label = 'Validation Top 5 accuracy')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Accuracy", size = 14)
    ax.set_title("Train vs Validation", size = 14, fontweight='bold')
    ax.legend()
    fig.set_figheight(6)
    fig.set_figwidth(16)
    fig.savefig("result/acc.png")
    
def plot_loss(loss, epoch):
    fig = plt.figure()
    plt.plot(loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    fig.savefig("result/loss.png")