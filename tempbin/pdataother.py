from support import *
length=300
mainpath=input('path')
datalistnamewithextension = []
datanplist=[]


# parameter for face detection
theDetectorProto = 'data/deploy.prototxt'  # model architect
theDetectorModel = 'data/res10_300x300_ssd_iter_140000.caffemodel'  # weight
theConfidence = 0.7  # minimum probability of detection to be a face
theFaceThreshold = 20  # minimum pixels to be a face
# parameter for drawing on the image
detector = cv2.dnn.readNetFromCaffe(theDetectorProto, theDetectorModel)
data=None

for subpath in listdir(mainpath):
    index=0
    for photopath in listdir(mainpath+'/'+subpath):
        index+=1
        try:
            frame = cv2.imread(mainpath+'/'+subpath+'/'+photopath)
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
                    face = frame[startY:endY, startX:endX]
                    data = standardphoto(length, frame[startY:endY, startX:endX].copy())
                    np.save('Dataset/ot1/'+str(subpath[:3]+'_'+str(index)), arr=BGR_RGB(data))
                    print('save in '+'Dataset/ot1/'+str(subpath[:3]+'_'+str(index)));
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large

            print('Succ in ' + str(mainpath+'/'+subpath+'/'+photopath))
        except:
            print('fail in ' +str(mainpath+'/'+subpath+'/'+photopath))

    #render(BGR_RGB(img))