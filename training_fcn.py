from network_arch.FCN import FullyConvNet, FullyConvNetYading

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.PyData import YaoZhuiDataSet
from torch.utils.data import DataLoader
from torchvision import transforms, utils

gpu_id = 0

tsfm = transforms.Compose([transforms.Resize((256,256)),
                           transforms.ToTensor()])

dataset = YaoZhuiDataSet(d_path = '/home/bcgpu0/Desktop/work_space_yhm/medstartup/data/temp/data/', l_path = '/home/bcgpu0/Desktop/work_space_yhm/medstartup/data/temp/label/', transforms=tsfm)

train_loader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2)

fcn = FullyConvNetYading().cuda(gpu_id)

print('training data length =', len(train_loader))
optimizer = optim.Adam(fcn.parameters())

def dice_loss(input, target):
  smooth = 1

  iflat = input.contiguous().view(-1)
  tflat = target.contiguous().view(-1)
  intersection = (iflat * tflat).sum()

  a_sum = torch.sum(iflat * iflat)

  b_sum = torch.sum(tflat * tflat)

  return 1.0 - (((intersection + smooth) /
                 (a_sum + b_sum - intersection + smooth)))

print('start training...')
fcn.train()
for epoch in range(1000):
  for i, data in enumerate(train_loader):
    l = len(train_loader)
    optimizer.zero_grad()

    input, labels = data
    print(input.size())

    batch_x, batch_y = Variable(input.type('torch.FloatTensor').cuda(gpu_id)), Variable(labels.type('torch.FloatTensor').cuda(gpu_id))


    outputs = fcn(batch_x)

    loss = dice_loss(outputs, batch_y)
    if i%5 == 0:
      print("epoch:", epoch)
      print(i,'/',l, loss)
    loss.backward()
    optimizer.step()

