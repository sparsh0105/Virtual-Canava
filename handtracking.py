import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

# create an object from our class hands

mphands = mp.solutions.hands
hands = mphands.Hands()

'''
mphands.Hands() consist of - (i) static_image_mode = False (by default)
sometimes it will detect sometimes it will track based on confidence levels 
if its True then all the time it will do the detection part which will make it quite slow 

(ii) max_num_hands = 2
(iii) min_detection_confidence = 0.5
(iv) min_tracking_confidence = 0.5

'''
mpDraw = mp.solutions.drawing_utils

ptime = 0   # previous time
ctime = 0   # current time

while True:
    success , img = capture.read()
    
    # send RGB image to hands object as it uses only RGB images
    imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # to check if something is detected or not
    #print(results.multi_hand_landmarks)

    # put a for loop to check if we have multiple hands or not and we have to extract them one by one

    if results.multi_hand_landmarks : 
        for handLms in results.multi_hand_landmarks:
            # get the id no. and landmark information (x,y coordinate)
            for id, lm in enumerate(handLms.landmark):
                #print(id , lm)
                h , w, c = img.shape
                cx , cy = int(lm.x * w) , int(lm.y * h)
                print(id , cx , cy)
                #if id == 8:
                    #cv.circle(img , (cx , cy) , 10 , (255,0,255) , thickness=-1)


            mpDraw.draw_landmarks(img , handLms , mphands.HAND_CONNECTIONS)     # not displaying rgb image we are dispalying normal image 

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img , str(int(fps)) , (10,50), cv.FONT_HERSHEY_PLAIN , 3 ,(255,0,0) ,thickness=2)

    cv.imshow('Webcam' , img)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break