from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(name, batch=False, size=62):
    file_path = name + '.txt'
    mean = np.array([[[0.485]], 
                     [[0.456]], 
                     [[0.406]]])

    std = np.array([[[0.229]], 
                    [[0.224]], 
                    [[0.225]]])
    if name == 'train':
        with open('train.txt', 'r') as f:
            content = f.read().split('\n')
            content = [i.split(' ') for i in content][:-1]
            
            data = np.zeros([batch,3,size,size])
            label = []
            sampel_content = random.sample(content, batch)
            
            for index, path in enumerate(sampel_content):
                img = imread(path[0])
                img = resize(img, (size,size))
                img = np.array(img)
                if img.ndim != 3:
                    img_rgb = np.zeros([3,size,size])
                    img_rgb[:] = img.reshape(1,size,size)
                    img_rgb = (img_rgb - mean)
                    data[index,:] = img_rgb
                else:
                    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                    img = (img - mean)
                    data[index,:] = img
                
                img_label = np.zeros(50)
                img_label[int(path[1])] = 1
                label.append(img_label)
    
        return data, np.array(label)
    
    else:
        file_path = name + '.txt'
        with open(file_path, 'r') as f:
            content = f.read().split('\n')          
            content = [i.split(' ') for i in content][:-1]
            
            if batch:
                data = np.zeros([batch,3,size,size])
                label = []
                sampel_content = random.sample(content, batch)
                
                for index, path in enumerate(sampel_content):
                    img = imread(path[0])
                    img = resize(img, (size,size))
                    img = np.array(img)
                    if img.ndim != 3:
                        img_rgb = np.zeros([3,size,size])
                        img_rgb[:] = img.reshape(1,size,size)
                        img_rgb = (img_rgb - mean)
                        data[index,:] = img_rgb
                    else:
                        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                        img = (img - mean)
                        data[index,:] = img

                    img_label = np.zeros(50)
                    img_label[int(path[1])] = 1
                    label.append(img_label)
            else:
                data = np.zeros([len(content),3,size,size])
                label = []

                for index, path in enumerate(content):
                    img = imread(path[0])
                    img = resize(img, (size,size))
                    img = np.array(img)
                    if img.ndim != 3:
                        img_rgb = np.zeros([3,size,size])
                        img_rgb[:] = img.reshape(1,size,size)
                        img_rgb = (img_rgb - mean)
                        data[index,:] = img_rgb
                    else:
                        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                        img = (img - mean)
                        data[index,:] = img

                    img_label = np.zeros(50)
                    img_label[int(path[1])] = 1
                    label.append(img_label)
        
        return data, np.array(label)
    
    
