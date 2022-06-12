import cv2
import sys
import numpy as np
from bb_intersection_over_union import bb_intersection_over_union
from time import time


#path to cascades
#cascPath = "cascade_920p_3019n_ckplus_corrected.xml"
cascPath = "cascade_3090p_3019n_15stages.xml"
#cascPath = "haar_cascade_all_9000p_4500n.xml"
buildPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
buildCascade = cv2.CascadeClassifier(buildPath)

def callBack(argument):
    pass

def oddCheck(num):
    if (num % 2) == 0:
        return num + 1
    else:
        return num

#Real-time face detection
#for webcam
cap = cv2.VideoCapture(0)

#create toolbars
cv2.namedWindow("Result")
cv2.createTrackbar("Alpha", "Result", 400, 1000, callBack)
cv2.createTrackbar("Neig", "Result", 8, 20, callBack)
cv2.createTrackbar("Blur", "Result", 1, 40, callBack)

switch = 'Compare'
cv2.createTrackbar(switch, 'Result',0,1,callBack)
loop_time = time()
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_value = 1 + (cv2.getTrackbarPos("Alpha", "Result") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Result")
    blur = cv2.getTrackbarPos("Blur", "Result")
    switch_status = cv2.getTrackbarPos(switch, "Result")
    #if blur is not odd change for odd number
    blur = oddCheck(blur)

    faces = faceCascade.detectMultiScale(gray, scale_value, neig)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (blur,blur), 0)
        if switch_status == 1:
            faces2 = buildCascade.detectMultiScale(gray, scale_value, neig)
            if len(faces2) == 0:
                pass
            else:
                for (x2, y2, w2, h2) in faces2:
                    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                    iou = bb_intersection_over_union([x,y,x+w,y+h], [x2,y2,x2+w2,y2+h2])
                    cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            pass

    cv2.rectangle(img, (10, 10), (10 + 150, 30 + 10), (0, 0, 0), -1)

    fps = np.round(1 / (time() - loop_time))
    loop_time = time()

    cv2.putText(img, "DPS: {:.4f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Result', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        cv2.imwrite('screenshots/{}.jpg'.format(loop_time), img)
cap.release()
cv2.destroyAllWindows()