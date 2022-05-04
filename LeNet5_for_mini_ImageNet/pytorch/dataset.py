import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class MiniImageNet(Dataset):
    def __init__(self, data_name, path, transform = None):
        super().__init__()
        self.path = path
        file_path = path + data_name + '.txt'
        with open(file_path, 'r') as f:
            content = f.read().split('\n')          
            self.data = [i.split(' ') for i in content][:-1]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)
        if img.ndim != 3:
            img_rgb = np.zeros([img.shape[0],img.shape[1],3])
            img_rgb[:] = img.reshape(img.shape[0],img.shape[1],1)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            
        return img_rgb, int(label)