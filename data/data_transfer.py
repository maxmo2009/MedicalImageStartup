import numpy as np
import nrrd
import scipy.misc

import matplotlib.pyplot as plt



i = 89
data_path = 'test_data.nrrd'
label_path = 'test_label.nrrd'

data, _h = nrrd.read(data_path)
label, _d = nrrd.read(label_path)

data = data/data.max()
number_of_data = data.shape[2]

for i in range(number_of_data):
  scipy.misc.imsave('temp/data/data_' + str(i) + ".jpeg", data[:,:,i])
  scipy.misc.imsave('temp/label/label_' + str(i) + ".jpeg", label[:, :, i])
