from linearRegression import *
import itertools
import cv2
import os
import numpy as np
from support import *
from linearRegression import *
import random
from dhash import *
td1=[]
for i in ['userdata/waito','userdata/jerry','userdata/bryan']:
    for j in os.listdir(i):
        td1.append(i+'/'+j)
td1=random.sample(td1,len(td1))
td2=list(itertools.combinations(td1,2))


same=[]
diff=[]
for x in td2:

    p=int(dhash(cv2.imread(x[0]),2**3))

    o=int(dhash(cv2.imread(x[1]),2**3))
    if x[0][:11]==x[1][:11]:
        same.append(disstance(p,o,2**3))
    else:
        diff.append(disstance(p,o,2**3))
    # for j in disstance(p,o,2**3):
    #     a.append(int(j))
    # a.append(200 if x[0][:3]==x[1][:3] else 700)
    # #print(a)
    # data.append(a)

print(len(same))
print(f' same {len(same)}   , max{max(same)} ,min {min(same)} ,avg {sum(same)/len(same)}')
print(f' diff {len(diff)}   , max{max(diff)} ,min {min(diff)} ,avg {sum(diff)/len(diff)}')
