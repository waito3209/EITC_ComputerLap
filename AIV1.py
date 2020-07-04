from support import *
import numpy as np
import matplotlib.pyplot as plt
traindatapath=input('traindatafile')
testdatapath=input('traindatafile')

datatrainlist = list(combinations(listdir(traindatapath), 2))
datatestlist = list(combinations(listdir(testdatapath), 2))
detail(datatrainlist)
traindata=[]
trainlabel=[]
testdata=[]
testlabel=[]
for i in list(datatrainlist):
    p1=np.load(traindatapath + '/'+i[1] )
    p2=np.load(traindatapath + '/'+i[0] )
    detail(traindata.append(np.reshape(np.append(p1,p2),300*300*2*3)))
    if i[1][:3]==i[0][:3]:
        trainlabel.append(1)
    else:
        trainlabel.append(0)
for i in list(datatestlist):
    p1=np.load(testdatapath + '/'+i[1] )
    p2=np.load(testdatapath + '/'+i[0] )
    detail(testdata.append(np.reshape(np.append(p1,p2),300*300*2*3)))
    if i[1][:3]==i[0][:3]:
        testlabel.append(1)
    else:
        testlabel.append(0)
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)
from sklearn.preprocessing import LabelBinarizer
lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)