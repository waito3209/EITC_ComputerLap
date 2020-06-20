import cv2

print('import cv2')
import sys

print('import sys')
import numpy as np

print('import numpy as np')
import time

print('import time')
import win32com.client  # 導入程式庫

print('win32com.client')
import matplotlib.pyplot as plt

print('import matplotlib.pyplot as plt')
import os
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
def BGR_RGB(img):
    for i in img:
        for y in i:
            temp1=y[0].copy()
            temp2 = y[2].copy()
            y[2]= temp1
            y[0]=temp2
    return img
def render(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()