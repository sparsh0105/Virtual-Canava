import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    '''
static_image_mode = False
max_num_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
'''
    def __init__(self , mode = False , maxHands = 2 , detectionCon = 0.5 , trackingCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self , img , draw = True):

        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks : 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms , self.mphands.HAND_CONNECTIONS)     

        return img
    def findPosition(self , img , handNo = 0 , draw = True):
         
         self.lmList = []
         if self.results.multi_hand_landmarks :
             
             myhand = self.results.multi_hand_landmarks[handNo]

             for id, lm in enumerate(myhand.landmark):
                    #print(id , lm)
                    h , w, c = img.shape
                    cx , cy = int(lm.x * w) , int(lm.y * h)

                    #print(id , cx , cy)
                    
                    self.lmList.append([id, cx , cy])
                    #if draw:
                        #if id == 8:
                            #cv.circle(img , (cx , cy) , 10 , (255,0,255) , thickness=-1)
         return self.lmList
    
    def fingersUp(self):
        fingers = []
    # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers


def main():
    ptime = 0  
    ctime = 0   
    capture = cv.VideoCapture(0)
    detector = handDetector()
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

if __name__ == '__main__':
    main()