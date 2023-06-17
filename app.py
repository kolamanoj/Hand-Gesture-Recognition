# import libraries

import cv2
import mediapipe as mp
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

# select some attributes
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# select the camera
cam = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
        success, image = cam.read()
        # Width, Height, Depth (Channels)
        imageWidth, imageHeight = image.shape[:2]
        if not success:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        results = hands.process(image)  # Pre-trained Deep Learning Model

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
                result = model .predict([data])
                print(result)
                font = cv2.FONT_HERSHEY_SIMPLEX

                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)  # Blue, Green, Red
                thickness = 2

                image = cv2.putText(
                    image, result[0], org, font, fontScale, color, thickness, cv2.LINE_4)

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
cam.release()
