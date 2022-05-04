from model import perceptron
import numpy as np

layer = [3780, 50]
dataset_dir = 'images_feature/'
model = perceptron(layer)
test_x, test_y = np.load(dataset_dir + 'test_x.npy'), np.load(dataset_dir + 'test_y.npy')
model.load('model_pretrained.npy')

top1 = model.predict(test_x, test_y, 1)
top5 = model.predict(test_x, test_y, 5)

print('Perceptron: top_1_accuracy %0.2f , top_5_accuracy %0.2f' % (top1, top5) + '%')