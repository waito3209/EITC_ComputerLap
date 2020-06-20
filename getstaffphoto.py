from support import *
length=300
cam = cv2.VideoCapture(0)  # 創建一個實例, 引用 opencv, 連接 usb webcam

if not cam.isOpened():  # webcam 連接# 功?
    print('Could not start camera!')
    sys.exit()  # 離開程式
# parameter for face detection
theDetectorProto = 'data/deploy.prototxt'  # model architect
theDetectorModel = 'data/res10_300x300_ssd_iter_140000.caffemodel'  # weight
theConfidence = 0.7  # minimum probability of detection to be a face
theFaceThreshold = 20  # minimum pixels to be a face
# parameter for drawing on the image
detector = cv2.dnn.readNetFromCaffe(theDetectorProto, theDetectorModel)
data=None



while input('takephoto')=='o':  # 重複做以下的程式碼

    hasFrame, frame = cam.read()  # get raw image
    cv2.imshow('image', frame)  # livestream

    # frame_RGB=BGR_RGB(frame)# 讀入一幀圖片
    # render(frame_RGB)# plt

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
            data=standardphoto(length,frame[startY:endY, startX:endX].copy())
            np.save('facedata/'+input('filename'),arr=BGR_RGB(data))
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < theFaceThreshold or fH < theFaceThreshold:
                continue

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    render(BGR_RGB(frame))  # convert BGR to RGB
cam.release()  # 釋放之前創建的實例
cv2.destroyAllWindows()  # 關閉所有創建的視窗