import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  
        self.conv1 = nn.Sequential(   
             nn.Conv2d(3, 6, 5, stride=1, padding=0),
             nn.Sigmoid(),
             nn.MaxPool2d(2, 2))
        
        self.conv2 = nn.Sequential(   
             nn.Conv2d(6, 16, 5, stride=1, padding=0),
             nn.Sigmoid(), 
             nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(29*29*16, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, 50))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
