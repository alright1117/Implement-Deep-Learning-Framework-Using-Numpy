import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import *
from dataset import MiniImageNet
from model import LeNet

args = get_args()

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

train_data = MiniImageNet('train', args.path, transform)
val_data = MiniImageNet('val', args.path, transform)

train_loader = DataLoader(dataset = train_data, batch_size = args.train_batch, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = val_data, batch_size = 1, shuffle=False, num_workers=0)

model = LeNet()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)

model = train_model(model, train_loader, valid_loader, device, criterion, optimizer, 200)

torch.save(model.state_dict(), 'model.pt')