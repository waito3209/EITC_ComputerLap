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
for i in ['wp','jp','bp']:
    for j in os.listdir(i):
        td1.append(i+'/'+j)
td1=random.sample(td1,len(td1))
td2=list(itertools.combinations(td1[:int(len(td1)/2)],2))
td3=list(itertools.combinations(td1[:-int(len(td1)/2)],2))

data=[]
for x in td2:
    a=[]
    p=int(dhash(cv2.imread(x[0]),2**3))

    o=int(dhash(cv2.imread(x[1]),2**3))
    for j in disstance(p,o,2**3):
        a.append(int(j))
    a.append(200 if x[0][:3]==x[1][:3] else 700)
    #print(a)
    data.append(a)
test=[]
for x in td3:
    a=[]
    p=int(dhash(cv2.imread(x[0]),2**3))

    o=int(dhash(cv2.imread(x[1]),2**3))
    for j in disstance(p,o,2**3):
        a.append(int(j))
    a.append(200 if x[0][:3]==x[1][:3] else 700)
    #print(a)
    test.append(a)
print(len(data))
print(len(test))
a=LinearRegression(data,'3003',0,False)
#a.var=list(np.load('004.npy'))
a.trainsimpleLR(500,0.00000001,False,testdata=test,showdot=10)
a.save()
a.report()
i=0
for x in a.trainreport.val[-1].deatilrecord:
    if x.abserror>(700-200)/2:
        i+=1
        #print(str(x))
print(i)