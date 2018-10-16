import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

train_data_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/WHS_MM/train_data/'
train_label_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/WHS_MM/train_label/'



def last_4chars(x):
  return (x[-24:])


label_list = os.listdir(train_label_path)
data_list = os.listdir(train_data_path)

label_list = sorted(label_list, key = last_4chars)
data_list = sorted(data_list, key = last_4chars)

image_list = []
gt_list = []

# label = np.pad(labo, pad_width = i, mode = 'constant', constant_values = 0)
# image = np.pad(img, pad_width= i, mode = 'constant', constant_values = 0)

for d, l in zip(data_list, label_list):
  d_p = train_data_path + d
  l_p = train_label_path + l


  img = nib.load(d_p).get_fdata()
  lab = nib.load(l_p).get_fdata()
# data is like [weight,height,index]
  img = np.swapaxes(img, 1, 2)
  img = np.swapaxes(img, 0, 1)

  img = img + abs(img.min())
  img = img / img.max()

  lab = np.swapaxes(lab, 1, 2)
  lab = np.swapaxes(lab, 0, 1)


  for i,j in zip(img, lab):
    if 420 in j:

      j = j == 420
      j = j.astype(int)



      i = resize(i, (256, 256), preserve_range=True)
      j = resize(j, (256, 256), preserve_range=True)

      j = np.pad(j, pad_width = 100, mode = 'constant', constant_values = 0)
      i = np.pad(i, pad_width= 100, mode = 'constant', constant_values = 0)

      image_list.append(i)
      gt_list.append(j)

    else:
      print('droped')
      continue

image_list = np.array(image_list)
gt_list = np.array(gt_list)

print(image_list.shape, image_list.max())
print(gt_list.shape, gt_list.max())

np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/WHS_MM/train_data.npy', image_list)
np.save('/home/bcgpu0/Desktop/work_space_yhm/dataset/WHS_MM/train_label.npy', gt_list)