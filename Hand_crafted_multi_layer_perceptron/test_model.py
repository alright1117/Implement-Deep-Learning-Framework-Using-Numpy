from nn import Model
from plot import cal_acc
import numpy as np

layer = [3780, 50]
lr = 0.1
input_size = 3780
hidden_size = 1000
output_size = 50

dataset_dir = '/home/ncku01/DL/hw1/images_feature/'
test_x, test_y = np.load(dataset_dir + 'test_x.npy'), np.load(dataset_dir + 'test_y.npy')

model = Model(input_size, hidden_size, output_size, lr)
model.load('model_pretrained.npy')
pred_y = model(test_x)

test_acc = cal_acc(pred_y, test_y)

print('Perceptron: top_1_accuracy %0.2f , top_5_accuracy %0.2f' % (test_acc[0], test_acc[1]) + '%')