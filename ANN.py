from tensorflow_core.python.keras.applications import MobileNetV2
import random
from support import *
import numpy as np
import matplotlib.pyplot as plt
import random as r
import PySimpleGUI as sg
import cv2
import numpy as np
from support import *
import tensorflow as tf
import copy
from dhash import dhash, disstance
from linearRegression import use
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
filelist=['userdata/jerry','userdata/waito','userdata/bryan']
#filelist=['jerry','waito','bryan']
length=800
name=input('name')
td1=[]
tt2=[]
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
    traindata.append(standardphoto(length,cv2.imread(str(i[0])+str(i[1])+str(i[2]))))
    #trainlabel.append(index_containing_substring(filelist,i[9:14]))
    trainlabel.append(index_containing_substring(filelist, i[0]))
testdata=[]
testlabel=[]
for i in tt2:
    testdata.append(standardphoto(length,cv2.imread(str(i[0])+str(i[1])+str(i[2]))))
    #trainlabel.append(index_containing_substring(filelist,i[9:14]))
    testlabel.append(index_containing_substring(filelist, i[0]))

traindata=np.asarray(traindata)

trainlabel=np.asarray(trainlabel)
testdata=np.asarray(testdata)
testlabel=np.asarray(testlabel)
testdata = testdata.astype("float32") / 255.0
traindata = traindata.astype("float32") / 255.0


testlabel=np.asarray(testlabel)
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)
from sklearn.preprocessing import LabelBinarizer
lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)
detail(train_labels_onehot)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
model = Sequential()
# first convolution: CONV => RELU => POOL
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(length,length,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second convolution: CONV => RELU => POOL
model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fully connected layer:  FC => RELU
model.add(Flatten())
model.add(Dense(20))
model.add(Activation("relu"))
model.add(Dense(10))
# Classifier:  softmax
model.add(Dense(3))
model.add(Activation("softmax"))
print( model.summary() )

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print("[INFO] training network...")
EPOCHS = 10
BS =8
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
print(classification_report(testlabel, predictions.argmax(axis=1),target_names=["waito","jerry",'bryan']))

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