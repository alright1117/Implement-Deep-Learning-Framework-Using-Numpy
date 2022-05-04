import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, losses
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import load_data
from model import make_model

path = '/home/alright/DL/hw4/'
val_data = load_data(path, 'val').batch(1)
test_data = load_data(path, 'test').batch(1)


model = make_model()
# load weights
model.load_weights("model.hdf5")
# Compile model (required to make predictions) 
opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
print("Created model and loaded weights from file")

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(val_data)
print('Validation accuracy:', scores[1])

scores = model.evaluate(test_data, verbose=0)
print('Test Accuracy:', scores[1])