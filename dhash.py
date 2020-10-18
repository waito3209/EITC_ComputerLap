import itertools
import cv2
import os
import numpy as np
from support import *
from linearRegression import *
import random





def dhash(image,hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



def disstance(d1, d2,hashSize=8):
    assert type(d1) == type(d2)
    hashSize=hashSize**2
    difference = 0
    a1 = "{0:b}".format(d1)
    a2 = "{0:b}".format(d2)
    a1=bin(int(d2))[2:].zfill(hashSize)
    a2  = bin(int(d1))[2:].zfill(hashSize)
    # print(a1)
    # print(a2)
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            difference+=1
        # else:
        #     difference.append('0')
    # for i in a1:
    #     difference.append(int(i))
    # for i in a2:
    #     difference.append(int(i))
    return difference
