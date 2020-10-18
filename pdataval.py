from support import *
length=600
mainpath=input('in')
outputpath=input('out')
datalistnamewithextension = []
datanplist=[]
detail(datalistnamewithextension)
for i in listdir(mainpath):
    print(i)
    datalistnamewithextension.append(str(i))
detail(datalistnamewithextension)

# parameter for face detection
theDetectorProto = 'data/deploy.prototxt'  # model architect
theDetectorModel = 'data/res10_300x300_ssd_iter_140000.caffemodel'  # weight
theConfidence = 0.7  # minimum probability of detection to be a face
theFaceThreshold = 20  # minimum pixels to be a face
# parameter for drawing on the image
detector = cv2.dnn.readNetFromCaffe(theDetectorProto, theDetectorModel)
data=None

for path in datalistnamewithextension:
        try:
            frame = cv2.imread(mainpath+'/'+path)
            # image_rgb =frame
            # mask = np.zeros(image_rgb.shape[:2], np.uint8)
            # bgdModel = np.zeros((1, 65), np.float64)
            # fgdModel = np.zeros((1, 65), np.float64)
            # rectangle=(1,1,frame.shape[0],frame.shape[1])
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
            # frame=image_rgb_nobg
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
                    face = frame[startY-10:endY+10, startX-10:endX+10]
                    image_rgb = face #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                    #data = standardphoto(length, frame[startY:endY, startX:endX].copy())
                    cv2.imwrite(outputpath+'/'+str(path),frame[startY:endY, startX:endX])
                    #(fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large

            print('Succ in ' + str(path))
        except:
            print('fail in ' +str(path))

    #render(BGR_RGB(img))