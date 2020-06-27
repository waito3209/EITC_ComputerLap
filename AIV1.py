from support import *
try:
    from __future__ import timemachine
except:
    pass

datalist = list(combinations(listdir(input('traindatafile')), 2))
detail(datalist)
for i in list(datalist):
    print(i)