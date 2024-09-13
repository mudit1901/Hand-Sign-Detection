import math
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cam = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 400
folder = "Data/B"
counter = 0

while True:
    ret, frame = cam.read()
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        if 'bbox' in hand:
            x, y, w, h = hand['bbox']
            h_frame, w_frame, _ = frame.shape

            # Ensure the coordinates are within the frame size
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(w_frame, x + w + offset)
            y2 = min(h_frame, y + h + offset)

            imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255

            imgCrop = frame[y1:y2, x1:x2]
            # Putting the Hand detecting Area into White Space
            imgCropShape = imgCrop.shape
            # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgsize / h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgsize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize

            else:
                k = imgsize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgsize-hCal)/2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)
        else:
            print("Bounding box not found in hand detection.")

    cv2.imshow("WebCam", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
