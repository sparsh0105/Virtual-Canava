import numpy as np
import cv2 as cv
import time 
import os
import requests 
import imutils
import handtrackingmodule as htm 

############################
brushthickness = 15
eraserthickness = 100
############################


folderpath = 'Header'
myList = os.listdir(folderpath)
#print(myList)

# import our images 
overlayList = []
for imgpath in myList:
     img = cv.imread(f'{folderpath}/{imgpath}')
     overlayList.append(img)
#print(len(overlayList))

header = overlayList[0]

drawcolour = (0,0,255)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)

xp,yp = (0,0)

imgCanvas = np.zeros((720,1280,3), np.uint8)
while True:
     # 1) import the image
     success , img = cap.read()
     img = cv.flip(img , 1)

     # 2) find hand landmarks
     img = detector.findHands(img)
     lmList = detector.findPosition(img , draw = False)
     if len(lmList) != 0 :
           #print(lmList)

           x1 , y1 = lmList[8][1:]           # tip point for index finger
           x2 , y2 = lmList[12][1:]          # tip point for middle finger
           #print(x1,y1,x2,y2)
           # 3) check which fingers are up

           fingers = detector.fingersUp()
           #print(fingers)

           # 4) if selection mode - 2 fingers are up
           if fingers[1] and fingers[2] :
                 xp,yp = (0,0)
                 
                 #checking for the click

                 if y1<125 :
                       if 250 < x1 < 450 :
                             header = overlayList[0]
                             drawcolour = (0,0,255)
                       elif 550 < x1 < 750 :
                             header = overlayList[1]
                             drawcolour = (0,255,0)
                       elif 800 < x1 < 950 :
                             header = overlayList[2]
                             drawcolour = (255,0,255)
                       elif 1055 < x1 < 1200 :
                             header = overlayList[3]  
                             drawcolour = (0,0,0)

                 cv.rectangle(img , (x1,y1) , (x2,y2), drawcolour, thickness=-1)       
                 #print('Selection mode')

           # 5) if drawing mode - 1 finger is up

           if fingers[1] and fingers[2] == False :
                 cv.circle(img , (x1,y1) , 15 ,drawcolour , thickness=-1)
                 #print('Drawing mode')

                 if xp == 0 and yp == 0 :
                       xp , yp = x1 , y1    # draw exactly at the same point wherever you are at

                 if drawcolour == (0,0,0):
                       cv.line(img , (xp,yp) , (x1,y1) , drawcolour , eraserthickness)
                       cv.line(imgCanvas , (xp,yp) , (x1,y1) , drawcolour , eraserthickness)
                 else: 
                       cv.line(img , (xp,yp) , (x1,y1) , drawcolour , brushthickness)
                       cv.line(imgCanvas , (xp,yp) , (x1,y1) , drawcolour , brushthickness)

                 xp , yp = x1 , y1


     imgGray = cv.cvtColor(imgCanvas , cv.COLOR_BGR2GRAY)
     _, imgInv = cv.threshold(imgGray , 50 ,255, cv.THRESH_BINARY_INV)
     imgInv = cv.cvtColor(imgInv , cv.COLOR_GRAY2BGR)

     # setting the header image
     img[0:125 , 0:1280] = header
     img = cv.addWeighted(img , 0.5 , imgCanvas , 0.5,0)
     cv.imshow('img' , img)
     cv.imshow('canvas' , imgCanvas)
     img = cv.bitwise_and(img , imgInv)
     img = cv.bitwise_or(img , imgCanvas)

     if cv.waitKey(20) & 0xFF == ord('d'):
            break