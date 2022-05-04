import tensorflow as tf
import os
import random
import numpy as np

def load_images(files):
    path = files[0]
    label = tf.cast(tf.strings.to_number(files[1]), tf.int32)
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    mean = np.array([[[0.485, 0.456 , 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    
    if len(image.shape) != 3:
        image = tf.expand_dims(image, 2)
        image = (image - mean) / std
        image = tf.repeat(image, 3, 2)
        image = tf.image.resize(image, (128, 128))
    else:
        image = (image - mean) / std
        image = tf.image.resize(image, (128, 128))
        
    return image, label

def load_data(path, data_name):
    file_path = path + data_name + '.txt'
    with open(file_path, 'r') as f:
        content = f.read().split('\n')          
        data = [i.split(' ') for i in content][:-1]
        data = [[os.path.join(path, i), j] for i, j in data]
    
    random.shuffle(data)
    files = tf.data.Dataset.from_tensor_slices(data)
    ds = files.map(load_images)
    return ds