from tensorflow_core.python.keras.applications import MobileNetV2

from support import *
import numpy as np
import matplotlib.pyplot as plt
import random as r
length=150
name=input('name')
traindatapath=input('traindatafile')
testdatapath=input('traindatafile')

datatrainlist = list(combinations(listdir(traindatapath), 2))
datatestlist = list(combinations(listdir(testdatapath), 2))
detail(datatrainlist)
traindata=[]
traindata1=[]
rawtraindata= {}
rawtestdata={}
trainlabel=[]
testdata=[]
testdata1=[]
testlabel=[]
ctd=0
cvd=0
cvs=0
cts=0
for i in listdir(traindatapath):
    rawtraindata[i]=standardphoto(length,cv2.imread(traindatapath + '/'+i ))
        #np.load(traindatapath + '/'+i )
   # rawtraindata[i] = cv2.cvtColor(rawtraindata[i], cv2.COLOR_BGR2GRAY)
for i in listdir(testdatapath):
    rawtestdata[i]=standardphoto(length,cv2.imread(testdatapath + '/'+i ))
   # rawtestdata[i] = cv2.cvtColor(rawtestdata[i], cv2.COLOR_BGR2GRAY)

for i in list(datatrainlist):
    # p1=np.load(traindatapath + '/'+i[1] )
    # p2=np.load(traindatapath + '/'+i[0] )
    # p1=cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
    # p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
    #traindata.append(np.reshape(np.append(p1,p2),length*length*2*3))





    if i[1][:3]==i[0][:3]:
        if cts< 500:
            p1 = rawtraindata[i[1]]
            p2 = rawtraindata[i[0]]
            cts+=1
            trainlabel.append(1)
            traindata.append(p1)
            traindata1.append(p2)
    else:
        if ctd<500:
            ctd+=1
            p1 = rawtraindata[i[1]]
            p2 = rawtraindata[i[0]]

            trainlabel.append(0)
            traindata.append(p1)
            traindata1.append(p2)
for i in datatestlist:
    # p11 = np.load(testdatapath + '/' + i[1])
    # p22 = np.load(testdatapath + '/' + i[0])
    # p22 = cv2.cvtColor(p22, cv2.COLOR_BGR2GRAY)
    # p11 = cv2.cvtColor(p11, cv2.COLOR_BGR2GRAY)
    if i[1][:3]==i[0][:3]:
        if cvs < 500:
            p11 = rawtestdata[i[1]]
            p22 = rawtestdata[i[0]]

            testlabel.append(1)
            cvs+=1
            testdata.append(p11)
            testdata1.append(p22)
    else:
        if cvd<500:
            cvd+=1
            p11 = rawtestdata[i[1]]
            p22 = rawtestdata[i[0]]

            testlabel.append(0)
            testdata.append(p11)
            testdata1.append(p22)
print(ctd)
print(cvd)

print(cts)
print(cvs)
traindata=np.asarray(traindata)
traindata1=np.asarray(traindata1)
trainlabel=np.asarray(trainlabel)
testdata=np.asarray(testdata)
testdata1=np.asarray(testdata1)

testlabel=np.asarray(testlabel)
traindata = traindata.astype("float32") / 255.0
testdata = testdata.astype("float32") / 255.0
traindata1 = traindata.astype("float32") / 255.0
testdata1 = testdata.astype("float32") / 255.0
try:
    detail(testdata)
    detail(testdata[1])
except:
    pass
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)

from sklearn.preprocessing import LabelBinarizer
lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)
detail(train_labels_onehot)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
IMG_SHAPE=(length,length,3 )
left_input = Input(shape=IMG_SHAPE, name='left_input')
left_1 = Conv2D(20, (5, 5), padding="same", input_shape=IMG_SHAPE)(left_input)
left_2=Activation("softmax")(left_1)
left_3=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(left_2)
left_4=Conv2D(50, (5, 5), padding="same")(left_3)
left_5=(Activation("softmax"))(left_4)
left_6=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(left_5)

# Fully connected layer:  FC => RELU
left_7=(Flatten())(left_6)

left_branch=Activation("softmax",name='left_branch')(left_7)


right_input = Input(shape=IMG_SHAPE, name='right_input')
right_1 = Conv2D(20, (5, 5), padding="same", input_shape=IMG_SHAPE)(right_input)
right_2=Activation("softmax")(right_1)
right_3=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(left_2)
right_4=Conv2D(50, (5, 5), padding="same")(right_3)
right_5=(Activation("softmax"))(right_4)
right_6=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(right_5)

# Fully connected layer:  FC => RELU
right_7=(Flatten())(right_6)

right_branch=Activation("softmax",name='right_branch')(right_7)

x = concatenate([left_branch, right_branch])
x1=Dense(10)(x)
predictions = Dense(2, activation='softmax', name='main_output')(x1)

model = Model(inputs=[left_input, right_input], outputs=predictions)
# model = Sequential()
# model.add(Dense(length/2, input_shape=(length*2*length*3,)))
# model.add(Dot(int(2**10)))
# model.add(Dense(int(2**4), activation='softmax'))
# model.add(Dense(int(2**4)))
# model.add(Dense(int(2**4), activation='relu'))
# model.add(Dense(int(2**4)))
# model.add(Dense(int(2**4), activation='relu'))
# model.add(Dense(int(2**4)))
# model.add(Dense(2, activation='softmax'))
print( model.summary() )

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print("[INFO] training network...")
EPOCHS = 10
BS =12
#from keras.utils import to_categorical
#H = model.fit(traindata, (train_labels_onehot), batch_size=BS,epochs=EPOCHS, verbose=1,validation_data=(testdata,  (test_labels_onehot)))
H=model.fit({'left_input': traindata, 'right_input': traindata1},
          {'main_output': train_labels_onehot},
          epochs=EPOCHS, batch_size=BS,validation_data=({'left_input': testdata, 'right_input': testdata1},test_labels_onehot))
try:
    model.save('AIV1_epoch20.h5')
    model.save(name + '.h5')
except:
    pass
try:
    print("[INFO] evaluating network...")
    [test_loss, test_acc] = model.evaluate({'left_input': testdata, 'right_input': testdata1}, test_labels_onehot, verbose=0)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
except :
    pass

predictions = model.predict({'left_input': testdata, 'right_input': testdata1})

from sklearn.metrics import classification_report
# classification_report(Ground true, prediction, class names)
print(classification_report(testlabel, predictions.argmax(axis=1),target_names=["1","0"]))

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