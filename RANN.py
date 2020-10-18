from tensorflow_core.python.keras.applications import MobileNetV2
import random
from support import *
import numpy as np
import matplotlib.pyplot as plt
import random as r
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
import PySimpleGUI as sg
import cv2
import numpy as np
from support import *
import tensorflow as tf
import copy
from dhash import dhash, disstance
from linearRegression import use
filelist=['userdata/jerry','userdata/waito','userdata/bryan']
#filelist=['jerry','waito','bryan']
length=800
name=input('name')
td1=[]
tt2=[]
inputmodel=tf.keras.models.load_model('ANN13.h5')
for i in filelist:
    temp=list(os.listdir(i))
    temp = random.sample(temp, len(temp))
    temp = random.sample(temp, len(temp))
    temp = random.sample(temp, len(temp))

    for j in range(len(temp)):
        temptemp=[]
        temptemp.append(i)
        temptemp.append('/')
        temptemp.append(temp[j])
        if j <len(temp)*0.7:
            td1.append(temptemp)
        else:
            tt2.append(temptemp)
traindata=[]
trainlabel=[]
for i in td1:
    temp=standardphoto(length,cv2.imread(str(i[0])+str(i[1])+str(i[2])))
    temp = temp.astype("float32") / 255.0
    n = []

    n.append(temp.reshape([-1, length, length, 3]))
    # face = np.expand_dims(face, -1)
    g = inputmodel.predict(n)
    g = np.asarray(g)
    traindata.append(g[0])
    #trainlabel.append(index_containing_substring(filelist,i[9:14]))
    trainlabel.append(1)
testdata=[]
testlabel=[]
for i in tt2:
    temp=standardphoto(length,cv2.imread(str(i[0])+str(i[1])+str(i[2])))
    temp = temp.astype("float32") / 255.0
    n = []

    n.append(temp.reshape([-1, length, length, 3]))
    # face = np.expand_dims(face, -1)
    g = inputmodel.predict(n)
    g = np.asarray(g)
    testdata.append(g[0])
    #trainlabel.append(index_containing_substring(filelist,i[9:14]))
    testlabel.append(1)

for i in os.listdir('userdata/train'):
    try:
        temp=standardphoto(length,cv2.imread('userdata/train/'+str(i)))
        temp = temp.astype("float32") / 255.0
        n = []

        n.append(temp.reshape([-1, length, length, 3]))
        # face = np.expand_dims(face, -1)
        g = inputmodel.predict(n)
        g = np.asarray(g)
        traindata.append(g[0])
        #trainlabel.append(index_containing_substring(filelist,i[9:14]))
        trainlabel.append(0)
    except:
        print(i)
for i in os.listdir('userdata/val'):
    try:
        temp=standardphoto(length,cv2.imread('userdata/val'+'/'+str(i)))
        temp = temp.astype("float32") / 255.0
        n = []

        n.append(temp.reshape([-1, length, length, 3]))
        # face = np.expand_dims(face, -1)
        g = inputmodel.predict(n)
        g=np.asarray(g)
        testdata.append(g[0])
        #trainlabel.append(index_containing_substring(filelist,i[9:14]))
        testlabel.append(0)
    except:
        print(i)

traindata=np.asarray(traindata)

trainlabel=np.asarray(trainlabel)
testdata=np.asarray(testdata)
testlabel=np.asarray(testlabel)
testdata = testdata.astype("float32")
traindata = traindata.astype("float32")


testlabel=np.asarray(testlabel)
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)
from sklearn.preprocessing import LabelBinarizer
lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)
detail(train_labels_onehot)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
model = Sequential()
# first convolution: CONV => RELU => POOL
model.add(Input(shape=(3,)))
model.add(Dense(3))
model.add(Activation("softmax"))
model.add(Dense(3))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Dense(3))
model.add(Activation("softmax"))
model.add(Dense(2))
model.add(Dense(2))
model.add(Activation("softmax"))
print( model.summary() )

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print("[INFO] training network...")
EPOCHS = 2000
BS =20
#from keras.utils import to_categorical
#H = model.fit(traindata, (train_labels_onehot), batch_size=BS,epochs=EPOCHS, verbose=1,validation_data=(testdata,  (test_labels_onehot)))
H=model.fit(traindata,
          trainlabel,
          epochs=EPOCHS, batch_size=BS,validation_data=(testdata,testlabel))
try:
    model.save('AIV1_epoch20.h5')

except:
    print("fail model.save('AIV1_epoch20.h5')")
try:
    model.save(name + '.h5')
except:
    print("fail model.save(name + '.h5')")
try:
    print("[INFO] evaluating network...")
    [test_loss, test_acc] = model.evaluate(testdata, testlabel, verbose=0)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
except:
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}.format(test_loss, test_acc")

predictions = model.predict(testdata)

from sklearn.metrics import classification_report
# classification_report(Ground true, prediction, class names)
print(classification_report(testlabel, predictions.argmax(axis=1),target_names=["include",'notinclud']),)

#Plot the Loss Curves
plt.figure(figsize=[12,12])
plt.plot(H.history['loss'],'r',linewidth=3.0)
plt.plot(H.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[12,12])
plt.plot(H.history['accuracy'],'r',linewidth=3.0)
plt.plot(H.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.show()


model.save(name+'.h5')