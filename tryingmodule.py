import cv2 as cv
import mediapipe as mp
import time
import handtrackingmodule as htm

ptime = 0  
ctime = 0   
capture = cv.VideoCapture(0)
detector = htm.handDetector()
while True:
    success , img = capture.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0 :
        print(lmList[8])

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img , str(int(fps)) , (10,50), cv.FONT_HERSHEY_PLAIN , 3 ,(255,0,0) ,thickness=2)

    cv.imshow('Webcam' , img)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break