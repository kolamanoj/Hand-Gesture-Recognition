# import libraries

import cv2 # convert  colors of image, frames, reading frames from camera, opent the camera
import mediapipe as mp # neural network analysis on image -> Hand (Identified) - Tracking
import numpy as np

# select some attributes
mp_hands = mp.solutions.hands # hands
mp_drawing = mp.solutions.drawing_utils # drawing utility to draw circles at landmarks
mp_drawing_styles = mp.solutions.drawing_styles # it is drawing a line in multi colors

# select the camera
cam = cv2.VideoCapture(0) # opening the camera
with mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
    ) as hands: # minimum confidence (detection, tracking-50%)
    while cam.isOpened(): # whether the camera is opened or not
        success, image = cam.read() #camera will read the frame in image
        # Width, Height, Depth (Channels)
        imageWidth, imageHeight = image.shape[:2]
        if not success: # camera is not working
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        results = hands.process(image)  # Pre-trained Deep Learning Model

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
        if results.multi_hand_landmarks: # hands are there in image (true)
            for hand_landmarks in results.multi_hand_landmarks: # left, right
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(
                ), mp_drawing_styles.get_default_hand_connections_style())

                data = []  # empty list
                for point in mp_hands.HandLandmark:  # traverse through 21 landmarks
                    # extract the landmark of respective point
                    normalizedLandmark = hand_landmarks.landmark[point]
                    data.append(normalizedLandmark.x)  # X - coordinate
                    data.append(normalizedLandmark.y)  # Y - coordinate
                    data.append(normalizedLandmark.z)  # Z - coordinate
                print(len(data))  # length of list = 63
                # print(data)
                data = str(data)
                data = data[1:-1]  # removing square brackets of list
                f = open('awesome.csv', 'a')
                f.write(str(data)+',awesome\n')
                f.close()

            cv2.imshow('Hand Tracking', image) #displaying the frame after the landmarks
            if cv2.waitKey(5) & 0xFF == 27: #exit or q or ctrl+c
                break
cam.release() #stop the camera
 