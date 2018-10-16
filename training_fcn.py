from network_arch.FCN import FullyConvNet, FullyConvNetYading
import torch.distributed as dist
from scipy import ndimage
import cv2
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.misc import imsave
from skimage import color
import numpy as np
from data.PyData import ISICDataSet, Ph2DataSet, JSTRDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils

gpu_id = 1

tsfm = transforms.Compose([transforms.Resize((256,256)),
                           # transforms.RandomVerticalFlip(0.5),
                           # transforms.RandomHorizontalFlip(0.5),
                           # transforms.RandomRotation((-45,45)),
                           transforms.ToTensor()])

# dataset = Ph2DataSet(transforms = tsfm)
# dataset = ISICDataSet(d_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/train_data/',
#                       l_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/train_label/',
#                       transforms = tsfm)
dataset = JSTRDataset(transforms=tsfm)


train_loader = DataLoader(dataset = dataset, batch_size = 8, shuffle = True, num_workers = 2)

fcn = FullyConvNetYading().cuda(gpu_id)
# fcn = nn.DataParallel(fcn)

print('training data length =', len(train_loader))
optimizer = optim.Adam(fcn.parameters())
# fcn.load_state_dict(torch.load('models/ph2/model_fcn_pretrain.pt'))
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
  # total_loss = torch.tensor(0.0).cuda(gpu_id)
  # permutation = torch.randperm(train_data.size()[0])
  for i, data in enumerate(train_loader):
    l = len(train_loader)
    optimizer.zero_grad()

    input, labels = data

    batch_x, batch_y = Variable(input.type('torch.FloatTensor').cuda(gpu_id)), Variable(labels.type('torch.FloatTensor').cuda(gpu_id))

    # batch_x = batch_x/batch_x.max()
    # batch_y = batch_y/batch_y.max()
    #
    # print('x max min', batch_x.max(), batch_x.min())
    # print('x max min', batch_y.max(), batch_y.min())


    outputs = fcn(batch_x)
    # print(outputs.shape)
    # batch_y = batch_y.view(-1)
    # outputs = outputs.view(-1)
    # print(outputs.size(), batch_y.size())
    loss = dice_loss(outputs, batch_y)
    # total_loss = total_loss + loss
    if i%5 == 0:
      print("epoch:", epoch)
      print(i,'/',l, loss)
    loss.backward()
    optimizer.step()
    # total_loss = total_loss + loss
  # if total_loss <= 0.0001:
  #   torch.save(unet.state_dict(), 'models/isic_2016/model_unet.pt')
  #   break
  torch.save(fcn.state_dict(), 'models/jstr/model_fcn.pt')

  # print('EPC:',epoch,'Loss', total_loss)