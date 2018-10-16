import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class FullyConvNet(nn.Module):
  def __init__(self):
    super(FullyConvNet, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 32, 3, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(128, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.conv5 = nn.Sequential(
      nn.Conv2d(256, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True)
    )

    self.fc1 = nn.Linear(8192, 2048)
    self.fc2 = nn.Linear(2048, 2048)
    self.fc3 = nn.Linear(2048, 2)

    self.max_pool2d = nn.MaxPool2d(2)
    self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)


  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    # print(num_features)
    return num_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.max_pool2d(x)

    x = self.conv2(x)
    x = self.max_pool2d(x)

    x = self.conv3(x)
    x = self.max_pool2d(x)

    x = self.conv4(x)
    x = self.max_pool2d(x)

    x = self.conv5(x)
    x = x.view(-1, self.num_flat_features(x))

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

class FullyConvNetYading(nn.Module):
  def __init__(self):
    super(FullyConvNetYading, self).__init__()
    self.conv12 = nn.Sequential(
      nn.Conv2d(1, 8, 5),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 16, 3),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(16, 32, 4),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(32, 64, 4),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.conv5 = nn.Sequential(
      nn.Conv2d(64, 64, 5),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5)
    )


    self.deconv1 = nn.Sequential(
      nn.ConvTranspose2d(64, 64, 5),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 4),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(32, 16, 4),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True)
    )

    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(16, 8, 3),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True)
    )


    self.deconv_o = nn.Sequential(
      nn.ConvTranspose2d(8, 1, 5),
      nn.Sigmoid()
    )


    self.max_pool2d = nn.MaxPool2d(2)
    self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)




  def forward(self, x):
    x = self.conv12(x)
    x = self.max_pool2d(x)


    x = self.conv3(x)
    x = self.max_pool2d(x)

    x = self.conv4(x)
    x = self.max_pool2d(x)

    x = self.conv5(x)

    x = self.deconv1(x)
    x = self.up_sampling(x)

    x = self.deconv2(x)
    x = self.up_sampling(x)


    x = self.deconv3(x)
    x = self.up_sampling(x)


    x = self.deconv4(x)

    x = self.deconv_o(x)




    return x





test = FullyConvNet()


pytorch_total_params = sum(p.numel() for p in test.parameters())
print(pytorch_total_params)