from dataset import MiniImageNet
from model import LeNet
from utils import get_args

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

args = get_args()

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
                                
val_data = MiniImageNet('val', args.path, transform)
val_loader = DataLoader(dataset = val_data, batch_size = 1, shuffle=False, num_workers=0)
test_data = MiniImageNet('test', args.path, transform)
test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle=False, num_workers=0)

model = LeNet()
model.load_state_dict(torch.load('model.pt'))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

val_acc = 0.0

model.eval()
for data, target in val_loader:

    data = data.to(device)
    target = target.to(device)

    output = model(data)
    val_acc += torch.sum(torch.argmax(output, axis=1) == target).cpu().numpy()

val_acc = val_acc/len(val_loader.sampler)

print('Val accuracy %f' % (val_acc))

test_acc = 0.0

for data, target in test_loader:

    data = data.to(device)
    target = target.to(device)

    output = model(data)
    test_acc += torch.sum(torch.argmax(output, axis=1) == target).cpu().numpy()

test_acc = test_acc/len(test_loader.sampler)

print('Test accuracy %f' % (test_acc))