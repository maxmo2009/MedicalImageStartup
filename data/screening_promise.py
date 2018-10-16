#promis12 dataset
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from os import listdir
import cv2
from skimage.transform import resize



def list_files1(directory, extension):
  return [f for f in listdir(directory) if f.endswith('.' + extension)]

def last_4chars(x):
  return (x[-20:])

label_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/PROMISE12/train_label/'
data_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/PROMISE12/train_data/'


data_list = list_files1(data_path,'mhd')
label_list = list_files1(label_path,'mhd')

label_list = sorted(label_list, key = last_4chars)
data_list = sorted(data_list, key = last_4chars)

image_list = []
gt_list = []

for l,d in zip(label_list,data_list):
  # print('Dealing with')
  # print(d)
  # print(l)
  # print('------')

  l_path = label_path + l
  d_path = data_path + d

  image = sitk.ReadImage(d_path)
  image = sitk.GetArrayFromImage(image)
  # image = image / image.max()
  label = sitk.ReadImage(l_path)
  label = sitk.GetArrayFromImage(label)

  for i, l in zip(image, label):
    print(i.shape)
    i = resize(i, (320, 320), preserve_range=True)
    l = resize(l, (320, 320), preserve_range=True)

    if l.max() == 0:
      print('Drop!!!')

      continue
    print('sucess')
    i = i/i.max()
    l = l/l.max()


    image_list.append(i)
    gt_list.append(l)

image_list = np.array(image_list)
gt_list = np.array(gt_list)

print(image_list.shape, image_list.max())
print(gt_list.shape, gt_list.max())

np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/PROMISE12/test_data_320by320.npy', image_list)
np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/PROMISE12/test_label_320by320.npy', gt_list)

