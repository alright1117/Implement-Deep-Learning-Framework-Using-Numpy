import argparse
import copy
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, default = '/home/alright/DL/hw4/')
    parser.add_argument("--iters", type = int, default = 1000)
    parser.add_argument("--train_batch", type = int, default = 1024)
    parser.add_argument("--val_batch", type = int, default = 128)
    parser.add_argument("--lr", type = float, default = 0.01)

    args = parser.parse_args([])
    
    return args

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
    
def plot_loss(train_loss, val_loss, path):
    fig = plt.figure()
    plt.plot(train_loss, color='seagreen', label = 'Training Loss')
    plt.plot(val_loss, color='red', label = 'Validation Loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    fig.savefig("result/" + path)
    

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epoch):

    model = model.to(device)
    # keeping-track-of-losses 
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    for _ in range(epoch):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        # training-the-model
        model.train()
        for data, target in tqdm(train_loader):
            # move-tensors-to-GPU 
            data = data.to(device)
            target = target.to(device)

            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)
            train_acc += torch.sum(torch.argmax(output, axis=1) == target).cpu().numpy()

        # validate-the-model
        model.eval()
        for data, target in valid_loader:

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = criterion(output, target)

            # update-average-validation-loss 
            valid_loss += loss.item() * data.size(0)
            valid_acc += torch.sum(torch.argmax(output, axis=1) == target).cpu().numpy()

        # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_acc = train_acc/len(train_loader.sampler)
        valid_acc = valid_acc/len(valid_loader.sampler)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            _, train_loss, valid_loss, train_acc, valid_acc))
    print('Best validation accuracy is %f' % (best_acc))
    model.load_state_dict(best_model_wts)
    plot_acc(train_accs, valid_accs, list(range(epoch)), 'acc.png')
    plot_loss(train_losses, valid_losses, 'loss.png')
    
    return model