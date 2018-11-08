# MedicalImageStartup
----
1. Transfer your data set in the following format: save images and labels in different folders in .jpg or .png files with a incremental digit as the index. 
For example:

  data\
  
  --1.jpg
  
  --2.jpg
  
  ...
  
  label\
  
  --1_label.jpg
  
  --2_label.jpg
  
  ...
----
2. Use training_fcn.py to train your model after you transfer the dataset. Change the path in line 17 in training_fcn.py to the folder   you store your lable and images. Then, you should be able to run the program.
  
  
  
