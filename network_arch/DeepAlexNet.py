import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAlexNet3D(nn.Module):
  def __init__(self):
    super(DeepAlexNet, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv4 = nn.Conv2d(128, 256, 3)
    self.conv5 = nn.Conv2d(256, 512, 3)
    self.conv6 = nn.Conv2d(512, 512, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(8192, 2048)
    self.fc2 = nn.Linear(2048, 2)
    self.dropout = nn.Dropout(0.5)



  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.relu(self.conv3(x))
    x = self.dropout(x)
    x = F.max_pool2d(F.relu(self.conv4(x)), 2)
    x = F.relu(self.conv5(x))
    x = F.max_pool2d(F.relu(self.conv6(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x


  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    # print(num_features)
    return num_features

class DeepAlexNet(nn.Module):
  def __init__(self):
    super(DeepAlexNet, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    self.conv1 = nn.Conv2d(1, 32, 3)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv4 = nn.Conv2d(128, 256, 3)
    self.conv5 = nn.Conv2d(256, 512, 3)
    self.conv6 = nn.Conv2d(512, 512, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(8192, 2048)
    self.fc2 = nn.Linear(2048, 2)



  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(F.relu(self.conv4(x)), 2)
    x = F.relu(self.conv5(x))
    x = F.max_pool2d(F.relu(self.conv6(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x


  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    # print(num_features)
    return num_features

test = DeepAlexNet()


pytorch_total_params = sum(p.numel() for p in test.parameters())
print(pytorch_total_params)