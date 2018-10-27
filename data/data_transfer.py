import numpy as np
import nrrd

import matplotlib.pyplot as plt



i = 89
data_path = 'test_data.nrrd'
label_path = 'test_label.nrrd'

data, _h = nrrd.read(data_path)
label, _d = nrrd.read(label_path)

data = data/data.max()
print(data.shape)
print(label[:,:,i].max())

plt.imshow(label[:,:,i])
plt.savefig('1.png')