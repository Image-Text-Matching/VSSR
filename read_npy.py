import numpy as np

# 读取npy文件
data = np.load('D:\study\FACTUAL/f30k/test_caps_synthesis_florence_bge_det.npy')
print(data.shape)
#(5000,185,1024)

# data = np.load('D:\study\FACTUAL/f30k/test_caps_synthesis_florence_entities_bge_det.npy')
# print(data[0])
# print(data.shape)
#（5000，13，1024）