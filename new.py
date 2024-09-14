import numpy as np
import cv2 as cv
import requests
import imutils
import os
import handtrackingmodule as htm  # Assuming this is your custom module

# Constants
brushthickness = 15
eraserthickness = 100

# Folder containing header images
folderpath = 'Header'
myList = os.listdir(folderpath)

# Import the images from the folder
overlayList = [cv.imread(f'{folderpath}/{imgpath}') for imgpath in myList]
header = overlayList[0]  # Default header image

drawcolour = (0, 0, 255)  # Initial color (Red for drawing)

# URL for phone's webcam (Replace this with your phone's IP)
url = "http://192.168.1.64:8080/shot.jpg"

# Create a hand detector instance
detector = htm.handDetector(detectionCon=0.85)

# Initialize previous points
xp, yp = 0, 0

# Blank canvas for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1) Import the image from the phone's webcam
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1280, height=720)
    img = cv.flip(img, 1)

    # 2) Find hand landmarks using your hand tracking module
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get coordinates of index and middle finger tips
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3) Check which fingers are up
        fingers = detector.fingersUp()

        # 4) Selection mode (Two fingers up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # Reset the previous points

            # Check for header click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawcolour = (0, 0, 255)  # Red
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawcolour = (0, 255, 0)  # Green
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawcolour = (255, 0, 255)  # Purple
                elif 1055 < x1 < 1200:
                    header = overlayList[3]
                    drawcolour = (0, 0, 0)  # Eraser

            cv.rectangle(img, (x1, y1), (x2, y2), drawcolour, thickness=-1)

        # 5) Drawing mode (One finger up)
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, drawcolour, thickness=-1)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Drawing or erasing
            if drawcolour == (0, 0, 0):  # Erasing
                cv.line(img, (xp, yp), (x1, y1), drawcolour, eraserthickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawcolour, eraserthickness)
            else:  # Drawing
                cv.line(img, (xp, yp), (x1, y1), drawcolour, brushthickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawcolour, brushthickness)

            xp, yp = x1, y1

    # Convert canvas to grayscale to create mask for drawing
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)

    # Combine original image with the canvas
    img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    # Set the header image
    img[0:125, 0:1280] = header

    # Display the image and canvas
    cv.imshow('img', img)
    cv.imshow('canvas', imgCanvas)

    # Break loop on 'd' key press
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Close all OpenCV windows
cv.destroyAllWindows()
