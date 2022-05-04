import numpy as np
import timeit
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, losses
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import load_data
from model import make_model

path = '/home/alright/DL/hw4/'
train_data = load_data(path, 'train').batch(1024)
val_data = load_data(path, 'val').batch(1)
test_data = load_data(path, 'test').batch(1)

model = make_model()

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

filepath="model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(train_data, epochs=200, callbacks=callbacks_list, validation_data=val_data)

fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig("results/acc.png")

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig("results/loss.png")

loss, acc = model.evaluate(test_data)
print('Test accuracy:', acc)