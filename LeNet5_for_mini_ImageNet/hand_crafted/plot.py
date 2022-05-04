import seaborn as sns
import matplotlib.pyplot as plt

def plot_acc(train_acc, val_acc, iters, path):
    sns.set_style("darkgrid")    

    fig = plt.figure()

    ax = sns.lineplot(x = iters, y=train_acc, color='seagreen', label = 'Train Top 1 accuracy')
    sns.lineplot(x = iters, y=val_acc, color='red', label = 'Validation Top 1 accuracy')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Accuracy", size = 14)
    ax.set_title("Train vs Validation", size = 14, fontweight='bold')
    ax.legend()
    fig.set_figheight(6)
    fig.set_figwidth(16)
    fig.savefig("result/" + path)
    
def plot_loss(loss, path):
    fig = plt.figure()
    plt.plot(loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    fig.savefig("result/" + path)