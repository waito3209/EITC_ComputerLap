import cv2
print('import cv2')

import sys
print('import sys')

import numpy as np
print('import numpy as np')

import time
print('import time')

import datetime
print('import datetime')

# import win32com.client  # 導入程式庫
# print('win32com.client')

import matplotlib.pyplot as plt
print('import matplotlib.pyplot as plt')

import os
print('import os')

from skimage import data, img_as_float
print('from skimage import data, img_as_float')


from skimage.metrics import structural_similarity as ssim
print('from skimage.metrics import structural_similarity as ssim')

from os import listdir
print('from os import listdir')

from os.path import isfile, isdir, join
print('from os.path import isfile, isdir, join')

import math
print('import math')

from itertools import permutations ,combinations
print('from itertools import permutations,combinations ')

def BGR_RGB(img):
    for i in img:
        for y in i:
            temp1=y[0].copy()
            temp2 = y[2].copy()
            y[2]= temp1
            y[0]=temp2
    return img
print('import def BGR_RGB(img):')

def render(img):
    plt.imshow(img)
    plt.show()
print('def render(img):')

def standardphoto(length,data):

    return cv2.resize(data, (length,length)
                      #, interpolation = cv2.INTER_AREA
    )
print('def standardphoto(length,data):')

def detail(something):
    print("---------------")
    print(type(something))
    try:
        print(something[0])
    except:
        print('**fail to print something[0]')
    try:
        print('shape : ', end='')
        print(something.shape)
    except:
        print('fail to print something.shape')
    try:
        print('len : ',end='')
        print(len(something))
    except:
        print('fail to print len(something)')
    now = datetime.datetime.now()
    print("Current date and time : " ,end='')
    print(now.strftime("%Y-%m-%d %H:%M:%S"))