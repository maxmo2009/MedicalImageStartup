#screening data for ph2
import numpy as np
from os import walk
import os
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc import imsave
from skimage.transform import resize

pre_size = (256, 256)

def last_4chars(x):
  return (x[-10:])

root_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/PH2Dataset/PH2 Dataset images/'

data_np_list = []
label_np_list = []
# label_list = os.listdir(label_path)
root_list = os.listdir(root_path)
root_list = sorted(root_list, key=last_4chars)
for i in root_list:
  Id = i[-3:]
  data_path = root_path + i + '/' + i + '_Dermoscopic_Image/' + i + '.bmp'
  label_path = root_path + i + '/' + i + '_lesion/' + i + '_lesion.bmp'
  img = ndimage.imread(data_path)
  lab = ndimage.imread(label_path, mode='L')

  img = img/img.max()
  lab = lab/lab.max()

  img = resize(img, pre_size, preserve_range= True)
  lab = resize(lab, pre_size, preserve_range=True)

  img = np.pad(img, pad_width=100, mode='constant', constant_values=0)
  lab = np.pad(lab, pad_width=100, mode='constant', constant_values=0)

  data_np_list.append(img)
  label_np_list.append(lab)


np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/PH2Dataset/processed_data/all_color_data.npy', data_np_list)
np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/PH2Dataset/processed_data/all_color_label.npy', label_np_list)