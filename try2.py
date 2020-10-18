import PySimpleGUI as sg
import cv2
import numpy as np
from support import *
import tensorflow as tf
import copy
from dhash import dhash, disstance
from linearRegression import use

theDetectorProto = 'data/deploy.prototxt'  # model architect
theDetectorModel = 'data/res10_300x300_ssd_iter_140000.caffemodel'
theConfidence = 0.7  # minimum probability of detection to be a face
theFaceThreshold = 20  # minimum pixels to be a face
detector = cv2.dnn.readNetFromCaffe(theDetectorProto, theDetectorModel)
data = None
length = 600
filename = 'userdata'
var = list(np.load('5005.npy'))
print(var)
# mymodel = tf.keras.models.load_model('n02.h5')
# mymodel.summary()
"""
Demo program that displays a webcam using OpenCV
"""


def main():
    sg.theme('Dark Blue 3')

    # define the window layout
    layout = [[sg.Text('WebCam Image', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Cap', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'),
               sg.Button('', size=(10, 1), font='Helvetica 14', button_color=('red', 'red'), key='light')
                  , ]]

    # create the window and show it without the plot
    window = sg.Window('Face Recognition', layout, location=(800, 400))  # location=(800, 400)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cam = cv2.VideoCapture(0)
    recording = True

    while True:
        check = False
        event, values = window.read(timeout=5)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Record':
            recording = True

        elif event == 'Cap':
            window.FindElement('light').Update(button_color=('green', 'green'))
            recording = False
            # img = np.full((480, 640), 255)  #np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300)
                                              , (104.0, 177.0, 123.0), swapRB=False, crop=False)
            detector.setInput(imageBlob)
            detections = detector.forward()
            (h, w) = frame.shape[:2]
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                prob = detections[0, 0, i, 2]
                # filter out weak detections
                if prob > theConfidence:
                    # compute the (x, y) coordinates of the bounding box for the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # extract the face ROI
                    face = copy.deepcopy(frame[startY:endY, startX:endX])

                    #face = frame[startY-10:endY+10, startX-10:endX+10]
                    #image_rgb = face #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('temp/temp.jpg',face)
                    # mask = np.zeros(image_rgb.shape[:2], np.uint8)
                    # bgdModel = np.zeros((1, 65), np.float64)
                    # fgdModel = np.zeros((1, 65), np.float64)
                    # rectangle=(1,1,abs(startX-endX),abs(startY-endY))
                    # cv2.grabCut(image_rgb,  # Our image
                    #             mask,  # The Mask
                    #             rectangle,  # Our rectangle
                    #             bgdModel,  # Temporary array for background
                    #             fgdModel,  # Temporary array for background
                    #             5,  # Number of iterations
                    #             cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle
                    # mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    #
                    # # Multiply image with new mask to subtract background
                    # image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
                    # data = copy.deepcopy(standardphoto(length, frame[startY:endY, startX:endX]))
                    datahash = dhash(face,2**3)
                    #render(image_rgb_nobg)
                    for j in os.listdir(filename):
                        counter = 0
                        diffcounter = 0
                        avg=[]
                        for k in os.listdir(filename + '/' + j):
                            target = cv2.imread(filename + '/' + j + '/' + k)
                            targethash = dhash(target,2**3)
                            distance = disstance(targethash, datahash,2**3)
                            print(f'{j} , is {distance}.')
                            avg.append(distance)
                            #score = use(distance, var)
                            #print(score)
                            # counter += 1
                        #     if abs(score - 20) > abs(score - 70):
                        #         diffcounter += 1
                        #         #print(j + '--------diff')
                        #     else:
                        #         #print(j + '---------same')
                        #         pass
                        # if float(diffcounter / counter) > 0.2:
                        #     print('it is not ' + j)
                        # else:
                        #     print('it is ' + j)
                        #print(float(diffcounter / counter))
                        print(f"{j}  avg __________________{sum(avg)/len(avg)}")
                    # mytensor = data.reshape((data.shape[0], length, length, 3))
                    # mytensor = mytensor.astype("float32") / 255.0
                    #
                    # for j in listdir('facedata'):
                    #     t=np.load('facedata/'+j)
                    #     t = standardphoto(length, t)
                    #     mytensor2 = t.reshape((t.shape[0], length, length, 3))
                    #     mytensor2 = mytensor.astype("float32") / 255.0
                    #
                    #
                    #     x = mymodel.predict({'left_input':data,'right_input':t})
                    #     #detail(x)
                    #     mymodel.predict
                    #     print(x)
                    #     pred = x.argmax(axis=1)
                    #     print(j, pred)

        if recording:
            ret, frame = cam.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

    cam.release()
    window.close()


main()
