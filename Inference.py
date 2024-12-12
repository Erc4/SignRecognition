#import libraries
import cv2 
import os
from ultralytics import YOLO

#import the class HandTracking
import HandsFollowing as ht

#initialize the camera
cap = cv2.VideoCapture(0)
# change the resolution
cap.set(3, 1280)
cap.set(4, 720)

#read the model
model = YOLO("ModeloNajera.pt")

#initialize the class HandTracking
detector = ht.HandTracking(Confdetection=0.2)


while True:
    #read the camera
    ret, img = cap.read()

    #get the hand landmarks
    img = detector.findHands(img, draw=False)

    #get the landmarks of hands
    list1, bBox, hand = detector.findPosition(img, NumHands=0, DrawPoints=False, DrawBox=False, color = [0, 255, 0])

    #if the hand is detected
    if hand == 1:
        #get the landmarks of the hand
        xmin, ymin, xmax, ymax = bBox

        #asign the margin to the box
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #made a cut of the image
        imgCrop = img[ymin:ymax, xmin:xmax]

        #resize the image
        imgCrop = cv2.resize(imgCrop, (640, 640), interpolation=cv2.INTER_CUBIC)

        #get the prediction of the model
        results = model.predict(imgCrop, conf = 0.55)

        #if there is a result
        if len(results) != 0:
            #les iterate
            for result in results:
                masks = result.masks
                coordenates = masks

                notations = results[0].plot()


        cv2.imshow("Image CUT", notations)


        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)



    cv2.imshow("Sign Lenaguage", img)
    t = cv2.waitKey(1)
    if t == 27:
        break


cap.release()
cv2.destroyAllWindows()