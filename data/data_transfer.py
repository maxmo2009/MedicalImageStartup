import numpy as np
import nrrd

import matplotlib.pyplot as plt



i = 89
data_path = 'test_data.nrrd'
label_path = 'test_label.nrrd'

data, _h = nrrd.read(data_path)
label, _d = nrrd.read(label_path)

data = data/data.max()
number_of_data = data.shape[2]

for i in range(number_of_data):
  plt.imshow(data[:,:,i])
  plt.savefig('temp/' + str(i) + "_data.png")

  plt.imshow(label[:, :, i])
  plt.savefig('temp/' + str(i) + "_label.png")