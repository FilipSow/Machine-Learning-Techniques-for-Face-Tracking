import cv2
import sys
import numpy as np
from bb_intersection_over_union import bb_intersection_over_union
from PIL import ImageGrab
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import itertools

cascPath = "cascade_3090p_3019n_15stages.xml"
buildPath = 'haarcascade_frontalface_default.xml'

#Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
buildCascade = cv2.CascadeClassifier(buildPath)
def callBack(argument):
    pass

#Real-time face detection
cap = cv2.VideoCapture('test_video.MP4')

#create toolbars
cv2.namedWindow("Result")
i = 0
num = 0
roi_hist = False
list_of_screens = [528,536,544,553, 797, 801, 805, 810]
loop_time = time()
while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_value = 1.05
    neig = 20
    if i < 1:
        faces = faceCascade.detectMultiScale(gray, scale_value, neig)
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faces2 = buildCascade.detectMultiScale(gray, scale_value, neig)
            if len(faces2) == 0:
                pass
            else:
                for (x2, y2, w2, h2) in faces2:
                    #cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                    iou = bb_intersection_over_union([x,y,x+w,y+h], [x2,y2,x2+w2,y2+h2])
                    if iou >=0.7:
                        roi = img[y:y + h, x: x + w]
                        #cv2.imwrite('roi.jpg', roi)
                        track_window = (x,y,w,h)

                        #change for hsv
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        #plt.hist(hsv_roi[:,:,0].ravel(), 180, [0, 180], density = True)
                        #plt.xlabel('Hue')
                        #plt.ylabel('Number of pixels')
                        #plt.title('Model histogram')
                        #plt.grid(True)
                        #plt.tight_layout()
                        #plt.savefig('figure1.png')
                        #plt.show()

                        #create mask to cover only object of intrest
                        #mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

                        #create histogram distribution
                        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

                        #normalize histogram to get value from 0 to 255
                        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                        #criteria for mean shift process
                        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                        i = 1
                    else:
                        pass

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if type(roi_hist) != bool:
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #use mean shift to move to object of intrest
        _, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow('roi', roi)
        #cv2.imwrite('hsv_roi.jpg', hsv_roi[:,:,0])
        cv2.imshow('Result', img2)
        cv2.imshow('dst', dst)
    fps = np.round(1 / (time() - loop_time))
    loop_time = time()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f') or num in list_of_screens:
        cv2.imwrite('screenshots/{}.jpg'.format(num), img)
        cv2.imwrite('screenshots/dst{}.jpg'.format(num), dst)
    num = num + 1
cap.release()
cv2.destroyAllWindows()

#plt.hist(roi_hist)
#plt.show()
