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
    traindata.append(np.reshape(np.append(p1,p2),300*300*2*3))
    if i[1][:3]==i[0][:3]:
        trainlabel.append(1)
    else:
        trainlabel.append(0)
for i in list(datatestlist):
    p1=np.load(testdatapath + '/'+i[1] )
    p2=np.load(testdatapath + '/'+i[0] )
    testdata.append(np.reshape(np.append(p1,p2),300*300*2*3))
    if i[1][:3]==i[0][:3]:
        testlabel.append(1)
    else:
        testlabel.append(0)
traindata=np.asarray(traindata)
trainlabel=np.asarray(trainlabel)
testdata=np.asarray(testdata)
testlabel=np.asarray(testlabel)
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)
from sklearn.preprocessing import LabelBinarizer
lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)


import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dense

model =tf.keras.Sequential()
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
#print( model.summary() )
#detail(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
print("[INFO] training network...")
EPOCHS = 20
BS = 256
H = model.fit(traindata, train_labels_onehot, batch_size=BS,epochs=EPOCHS, verbose=1,validation_data=(testdata, test_labels_onehot))

print("[INFO] evaluating network...")
[test_loss, test_acc] = model.evaluate(testdata, test_labels_onehot, verbose=0)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

predictions = model.predict(testdata)
from sklearn.metrics import classification_report
# classification_report(Ground true, prediction, class names)
print(classification_report(testlabel, predictions.argmax(axis=1),target_names=(1,0)))

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


model.save('mnist_fc_epoch20.h5')