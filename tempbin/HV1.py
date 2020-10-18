from support import *
import numpy as np
import matplotlib.pyplot as plt
import random as r

length = 300
name = input('name')
traindatapath = input('traindatafile')
testdatapath = input('traindatafile')


def dhash(image):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    hashSize=8
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


datatrainlist = list(combinations(listdir(traindatapath), 2))
datatestlist = list(combinations(listdir(testdatapath), 2))
#detail(datatrainlist)
traindata = []
rawtraindata = {}
rawtestdata = {}
trainlabel = []
testdata = []
testlabel = []
for i in listdir(traindatapath):
    temp= np.load(traindatapath + '/' + i)
    rawtraindata[i] = dhash(temp)
    print(rawtraindata[i])

    # rawtraindata[i] = cv2.cvtColor(rawtraindata[i], cv2.COLOR_BGR2GRAY)
for i in listdir(testdatapath):
    temp = np.load(testdatapath + '/' + i)
    rawtestdata[i] = dhash(temp)
    # rawtraindata[i] = cv2.cvtColor(rawtraindata[i], cv2.COLOR_BGR2GRAY)

for i in list(datatrainlist):
    # p1=np.load(traindatapath + '/'+i[1] )
    # p2=np.load(traindatapath + '/'+i[0] )
    # p1=cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
    # p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
    # traindata.append(np.reshape(np.append(p1,p2),length*length*2*3))

    if i[1][:3] == i[0][:3]:

        p1 = rawtraindata[i[1]]
        p1a=np.asarray(list(str(int(p1))))
        p2 = rawtraindata[i[0]]
        p2a = np.asarray(list(str(int(p2))))
        trainlabel.append(1)
        traindata.append(np.append(p1a, p2a))
    else:
        if r.random() > 0.7:
            p1 = rawtraindata[i[1]]
            p1a = np.asarray(list(str(int(p1))))
            p2 = rawtraindata[i[0]]
            p2a = np.asarray(list(str(int(p2))))

            trainlabel.append(0)
            traindata.append(np.append(p1a, p2a))
for i in list(datatestlist):
    # p11 = np.load(testdatapath + '/' + i[1])
    # p22 = np.load(testdatapath + '/' + i[0])
    # p22 = cv2.cvtColor(p22, cv2.COLOR_BGR2GRAY)
    # p11 = cv2.cvtColor(p11, cv2.COLOR_BGR2GRAY)
    if i[1][:3] == i[0][:3]:
        p11 = rawtestdata[i[1]]
        p22 = rawtestdata[i[0]]

        p1a = np.asarray(list(str(int(p11))))

        p2a = np.asarray(list(str(int(p22))))
        testlabel.append(1)

        testdata.append(np.append(p1a, p2a))
    else:

        p11 = rawtestdata[i[1]]
        p22 = rawtestdata[i[0]]
        p1a = np.asarray(list(str(int(p11))))

        p2a = np.asarray(list(str(int(p22))))
        testlabel.append(0)
        testdata.append(np.append(p1a, p2a))
traindata = np.asarray(traindata)
#traindata=traindata.astype('float64')
trainlabel = np.asarray(trainlabel)

testdata = np.asarray(testdata)
#testdata=testdata.astype('float64')
testlabel = np.asarray(testlabel)
print('Training data shape : ', traindata.shape, trainlabel.shape)
print('Testing data shape : ', testdata.shape, testlabel.shape)
from sklearn.preprocessing import LabelBinarizer

lblbin = LabelBinarizer()
train_labels_onehot = lblbin.fit_transform(trainlabel)
test_labels_onehot = lblbin.transform(testlabel)
detail(train_labels_onehot)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

model = Sequential()
model.add(Dense(2, input_shape=(2,10,)))
for i in range(6):
    model.add(Dense(int(2+i)))
for i in range(6):
    model.add(Dense(int(2+6-i)))
model.add(Dense(int(2 ** 2), activation='relu'))
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[INFO] training network...")
EPOCHS = 100
BS = 5
# from keras.utils import to_categorical
H = model.fit(traindata, (train_labels_onehot), batch_size=BS, epochs=EPOCHS, verbose=1,
              validation_data=(testdata, (test_labels_onehot)))

print("[INFO] evaluating network...")
[test_loss, test_acc] = model.evaluate(testdata, test_labels_onehot, verbose=0)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
model.save('AIV1_epoch20.h5')

predictions = model.predict(testdata)

from sklearn.metrics import classification_report

# classification_report(Ground true, prediction, class names)
print(classification_report(testlabel, predictions.argmax(axis=1), target_names=["1", "0"]))

# Plot the Loss Curves
plt.figure(figsize=[12, 12])
plt.plot(H.history['loss'], 'r', linewidth=3.0)
plt.plot(H.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Plot the Accuracy Curves
plt.figure(figsize=[12, 12])
plt.plot(H.history['accuracy'], 'r', linewidth=3.0)
plt.plot(H.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.show()

model.save(name + '.h5')
