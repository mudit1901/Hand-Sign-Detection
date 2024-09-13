import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
from imutils.video import VideoStream
import time

# Initialize the HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
offset = 20
imgsize = 400

labels = ["A", "B", "C"]

# Streamlit title and description
st.title("Real-Time Hand Sign Detection")
st.text("This application detects hand signs in real-time using a webcam feed.")

# Video Stream
video_stream = VideoStream(src=0).start()

stop = st.button("Stop Webcam")

# Display the video stream
frame_placeholder = st.empty()

while True:
    if stop:
        video_stream.stop()
        break
    # Reading frame By Frame
    frame = video_stream.read()

    # Detect hands
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        if 'bbox' in hand:
            x, y, w, h = hand['bbox']

            # Create a white image for placing the hand cropped region
            imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(frame.shape[1], x + w +
                         offset), min(frame.shape[0], y + h + offset)

            # Crop the hand region
            imgCrop = frame[y1:y2, x1:x2]

            # Resize and place the cropped hand in the white image
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgsize / h
                wCal = int(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                wGap = (imgsize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgsize / w
                hCal = int(k * h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                hGap = (imgsize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Perform classification on the hand image
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.putText(frame, labels[index], (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)

    # Displaying the Output
    frame_placeholder.image(frame, channels="BGR")

video_stream.stop()
cv2.destroyAllWindows()
