from HS import HS

import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import time

import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_root = 'shirts4_v1/'
    frame_optical = np.zeros(59)

    
    for i in tqdm(range(1, 60)):
        im1 = cv2.imread(os.path.join(img_root, str(i).zfill(4) + '.png'))
        im2 = cv2.imread(os.path.join(img_root, str(i + 1).zfill(4) + '.png'))## 4.2ms
        # print(im1.shape)
        # exit(123)
        
        u, v = HS(im1, im2)# 0.41~0.44s
        
        # print(np.sum(u ** 2))
        # exit(123)

        frame_optical[i-1] = np.sum((u ** 2 + v ** 2) ** 0.5)# 0.7ms
        
    
    print(frame_optical)

    x = range(1, 60)
    plt.plot(x, frame_optical, 'b*--', alpha=0.5, linewidth=1, label='flow')
    plt.legend()  #显示上面的label
    plt.xlabel('image idx') #x_label
    plt.ylabel('frame optical')#y_label
    
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.show()