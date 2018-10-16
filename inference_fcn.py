from medical_tools.test_tools import *
from network_arch.FCN import FullyConvNetYading
import matplotlib.pyplot as plt
import os
from medpy.metric.binary import dc
from medpy.metric.binary import jc
from medpy.metric.binary import sensitivity
from medpy.metric.binary import specificity
from medpy.metric.binary import precision
from medpy.metric.binary import recall
from skimage.transform import rotate
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from skimage.transform import resize
import cv2

rotation_degree = 90
gpu_id = 0
data_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/test_data/'
label_path = '/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/test_label/'

def last_4chars(x):
  return (x[-24:])

label_list = os.listdir(label_path)
data_list = os.listdir(data_path)

label_list = sorted(label_list,key = last_4chars)
data_list = sorted(data_list, key = last_4chars)



fcn = FullyConvNetYading().cuda(gpu_id)
fcn.load_state_dict(torch.load('models/isic_2016/model_fcn.pt'))

dc_list = []
jc_list = []
sens_list = []
spec_list = []
pres_list = []
recall_list = []

success = 0
fcn.eval()
for img, lab in zip(data_list, label_list):

  print(img)
  print(lab)

  test_data_path = data_path + img
  img = ndimage.imread(test_data_path)
  img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  img = rotate(img, rotation_degree,preserve_range=True)
  img = img.astype(np.float32) / 255.0
  img = np.swapaxes(img,2,1)
  img = np.swapaxes(img, 1, 0)
  img = img[np.newaxis,:,:,:]

  # shape = (2048, 2048)  # matrix size
  # dtype = np.dtype('>u2')
  #
  # fid = open(test_data_path, 'rb')
  # data = np.fromfile(fid, dtype)
  # img = data.reshape(shape)
  # img = img/img.max()
  # img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  #
  # img = img[np.newaxis,np.newaxis,:,:]


  # img = img[np.newaxis,np.newaxis,:, :]

  test_label_path = label_path + lab
  lab = ndimage.imread(test_label_path)
  lab = cv2.resize(lab, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  lab = lab.astype(np.float32) / 255.0
  lab = rotate(lab, rotation_degree, preserve_range=True)

  print(img.shape, lab.shape)
  print(img.max(), lab.max())

  # non_zero_count = np.count_nonzero(lab)
  # ratio = non_zero_count / (lab.shape[0] * lab.shape[1])
  # print(ratio)


  input = Variable(torch.from_numpy(img).type('torch.FloatTensor').cuda(gpu_id))
  res = fcn(input).detach().cpu().numpy()
  # res =res/res.max()
  # res = (res-1)/res.max()


  ret, res = cv2.threshold(res[0,0,:,:], 0.8, 1, cv2.THRESH_BINARY)
  # res = res[0,0,:,:]
  dice = dc(lab, res)
  jaccard = jc(lab, res)
  sens = sensitivity(lab, res)
  spec = specificity(lab, res)
  pres = precision(lab, res)
  rec = recall(lab, res)

  print('dice:', dice)
  print('jaccad:', jaccard)
  print('sensitivity:', sens)
  print('precision:', pres)

  dc_list.append(dice)
  jc_list.append(jaccard)
  sens_list.append(sens)
  spec_list.append(spec)
  pres_list.append(pres)
  recall_list.append(rec)

  print('average dice:', sum(dc_list) / len(dc_list))
  print('average jaccard:', sum(jc_list) / len(jc_list))
  print('average sensitivity:', sum(sens_list) / len(sens_list))
  print('average specificity:', sum(spec_list) / len(spec_list))
  print('average precision:', sum(pres_list) / len(pres_list))
  print('average recall:', sum(recall_list) / len(recall_list))

  # plt.imshow(res + lab)
  # plt.show()
  # img = rgb2gray(img[0, : , :, :])
  plt.imsave('/home/bcgpu0/Desktop/work_space_yhm/dataset/isic_2016/res/' + str(success) + '.png', 0.5*res  + lab, cmap='gray')
  print(success)
  success = success + 1