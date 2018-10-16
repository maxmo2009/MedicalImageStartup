import numpy as np
from scipy import ndimage
import cv2
from scipy.misc import imsave
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from skimage import color
from skimage import io
from skimage.transform import resize

class ISICDataSet(Dataset):

  def __init__(self, d_path, l_path, transforms = None):
    self.data_path = d_path
    self.label_path = l_path
    self.transform = transforms

  def __getitem__(self, index):
    def last_4chars(x):
      return (x[-24:])

    data_file = os.listdir(self.data_path)
    label_file = os.listdir(self.label_path)

    label_list = sorted(label_file, key=last_4chars)
    data_list = sorted(data_file, key=last_4chars)

    img = Image.open(self.data_path + data_list[index])
    lab = Image.open(self.label_path + label_list[index])


    if self.transform:
      return self.transform(img), self.transform(lab)

    return img, lab

  def __len__(self):
    return len(os.listdir(self.data_path))


class Ph2DataSet(Dataset):

  def __init__(self, root_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/PH2Dataset/PH2 Dataset images/', transforms = None):
    def last_4chars(x):
      return (x[-10:])
    self.rootpath = root_path
    self.root_list = os.listdir(root_path)
    self.root_list = sorted(self.root_list, key=last_4chars)
    self.transform = transforms

  def __getitem__(self, index):
    i = self.root_list[index]
    Id = i[-3:]

    data_path = self.rootpath + i + '/' + i + '_Dermoscopic_Image/' + i + '.bmp'
    label_path = self.rootpath + i + '/' + i + '_lesion/' + i + '_lesion.bmp'

    img = Image.open(data_path)
    lab = Image.open(label_path)


    if self.transform:
      return self.transform(img), self.transform(lab)

    return img, lab

  def __len__(self):
    return len(self.root_list)

class PROMISEDataSet(Dataset):
  def __init__(self, d_path, l_path, transforms=None):
    self.data_path = d_path
    self.label_path = l_path
    self.transform = transforms
  def __getitem__(self, item):
    # im = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))
    pass
  def __len__(self):
    pass


class PatchDataset(Dataset):

  def __init__(self, p_path, v_path):

    self.patch_path = p_path
    self.vec_path = v_path
    self.data_len = len(os.listdir(self.patch_path))


  def __len__(self):
    return self.data_len

  def __getitem__(self, index):
    vec_array = np.load(self.vec_path)

    return np.load(self.patch_path + str(index) + '.npy'), vec_array[index]


class JSTRDataset(Dataset):
  def __init__(self, p_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/train_data/', l_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/train_label/', transforms=None):

    self.data_path = p_path
    self.label_path = l_path

    self.data_files = os.listdir(p_path)
    self.label_files = os.listdir(l_path)
    self.transform = transforms
    self.data_len = len(os.listdir(self.data_path))

  def __len__(self):
    return self.data_len

  def __getitem__(self, index):
    def last_4chars(x):
      return (x[-10:])

    data_list = sorted(self.data_files, key=last_4chars)
    label_list = sorted(self.label_files, key=last_4chars)

    shape = (2048, 2048)  # matrix size
    dtype = np.dtype('>u2')

    fid = open(self.data_path + data_list[index], 'rb')
    data = np.fromfile(fid, dtype)
    img = data.reshape(shape)
    lab = ndimage.imread(self.label_path + label_list[index])

    img = img / img.max()
    lab = lab / lab.max()

    img = resize(img, (256, 256), preserve_range=True)
    lab = resize(lab, (256, 256), preserve_range=True)

    img = Image.fromarray(img)
    lab = Image.fromarray(lab)

    if self.transform:
      return self.transform(img), self.transform(lab)

    return img, lab











