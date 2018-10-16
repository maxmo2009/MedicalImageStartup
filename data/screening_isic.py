#screening data for isic dataset
import numpy as np
from os import walk
import os
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc import imsave

label_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/test_label/'
data_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/test_data/'


def last_4chars(x):
  return (x[-24:])


label_list = os.listdir(label_path)
data_list = os.listdir(data_path)


id = 0
progress = 0

data_np_list = []
label_np_list = []

r = 0.08

def pad_image_label(img, labo, r = 0.08):
  for i in range(5000):
    # print(i)
    label = np.pad(labo, pad_width=i, mode= 'constant', constant_values=0)
    non_zero_count = np.count_nonzero(label)
    ratio = non_zero_count / (label.shape[0] * label.shape[1])
    # print(ratio)
    if ratio >= r:
      continue
    else:
      label = np.pad(labo, pad_width = i, mode = 'constant', constant_values = 0)
      image = np.pad(img, pad_width= i, mode = 'constant', constant_values = 0)
      return image, label
  # return image, label



for l,d in zip(label_list,data_list):
  print(d)
  print(l)

  l_path = label_path + l
  d_path = data_path + d


  segmentation = ndimage.imread(l_path)
  data = ndimage.imread(d_path,mode='L')
  non_zero_count = np.count_nonzero(segmentation)
  ratio = non_zero_count/(segmentation.shape[0]*segmentation.shape[1])

  # print(non_zero_count,'|',segmentation.shape,'|',ratio)

  if ratio < r:
    # segmentation = ndimage.gaussian_filter(segmentation, sigma=3)
    # ret, segmentation = cv2.threshold(segmentation, 0.05, 1, cv2.THRESH_BINARY)
    data = cv2.resize(data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    segmentation = cv2.resize(segmentation, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    data = data/data.max()
    segmentation = segmentation/segmentation.max()


    print(data.shape)
    print(data.max())
    print(segmentation.shape)
    print(segmentation.max())


    data_np_list.append(data)
    label_np_list.append(segmentation)

    # plt.imsave('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_data/' + str(id) +'.png', data)
    # plt.imsave('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_label/seg' + str(id) + '.png', segmentation)

    id = id + 1

  elif ratio >= r:

    # label = ndimage.gaussian_filter(segmentation, sigma=4)
    # ret, label = cv2.threshold(label, 0.1, 1, cv2.THRESH_BINARY)
    data = cv2.resize(data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    segmentation = cv2.resize(segmentation, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    data, segmentation = pad_image_label(data, segmentation)

    data = cv2.resize(data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    segmentation = cv2.resize(segmentation, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    data = data / data.max()
    segmentation = segmentation / segmentation.max()

    print(data.shape)
    print(data.max())
    print(segmentation.shape)
    print(segmentation.max())

    # print(data.max(),data.shape)

    data_np_list.append(data)
    label_np_list.append(segmentation)

    # plt.imsave('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_data/' + str(id) + '.png', data)
    # plt.imsave('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_label/seg' + str(id) + '.png', segmentation)

    id = id + 1




  progress = progress + 1
  print(progress ,'/',len(label_list))

print('final format: ')
print(np.asarray(data_np_list).shape)
print(np.asarray(label_np_list).shape)

print(np.asarray(data_np_list).max())
print(np.asarray(label_np_list).max())

#

np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_data_non_smooth.npy', np.asarray(data_np_list))
np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/processed_data_padding/test_label_non_smooth.npy', np.asarray(label_np_list))

