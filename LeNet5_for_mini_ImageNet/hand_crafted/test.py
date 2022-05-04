import numpy as np
import argparse

from dataset import load_data
from model import LeNet, LeNet_plus

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default = 'LeNet')
args = parser.parse_args()

if args.model == 'LeNet':
    model = LeNet()
    model.load('LeNet.npy')
    size = 128
else:
    model = LeNet_plus()
    model.load('LeNet_plus.npy')
    size = 126

test_x, test_y = load_data('val', size=size)
test_y = np.argmax(test_y, axis=1).reshape(((len(test_y),1)))
Y_pred = model(test_x)
Y_pred = Y_pred.argsort(axis=1)[:,-1:]
test_acc = (Y_pred == test_y).mean()

print("test accuracy: %s" % (test_acc))