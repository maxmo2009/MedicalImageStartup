#screening data for lung mentory
import numpy as np
from os import walk
import os
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc import imsave
from skimage.transform import resize
from scipy import ndimage

pre_size = (256, 256)
train_data_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/data/'
train_label_heart_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/label/heart/'


def last_4chars(x):
  return (x[-14:])

label_list = os.listdir(train_label_heart_path)
data_list = os.listdir(train_data_path)

label_list = sorted(label_list, key = last_4chars)
data_list = sorted(data_list, key = last_4chars)

data_np_list = []
label_np_list = []
for d, l in zip(data_list, label_list):
  d_p = train_data_path + d
  l_p = train_label_heart_path + l

  shape = (2048, 2048)  # matrix size
  dtype = np.dtype('>u2')

  fid = open(d_p, 'rb')
  data = np.fromfile(fid, dtype)
  img = data.reshape(shape)
  lab = ndimage.imread(l_p)

  img = img / img.max()
  lab = lab / lab.max()

  img = resize(img, pre_size, preserve_range=True)
  lab = resize(lab, pre_size, preserve_range=True)

  img = np.pad(img, pad_width=100, mode='constant', constant_values=0)
  lab = np.pad(lab, pad_width=100, mode='constant', constant_values=0)

  plt.imshow(img,  cmap='gray')
  plt.show()

  data_np_list.append(img)
  label_np_list.append(lab)

# np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/processed_data/all_heart_data.npy', data_np_list)
# np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/JSTR/processed_data/all_heart_label.npy', label_np_list)
