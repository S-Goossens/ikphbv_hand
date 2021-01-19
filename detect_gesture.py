from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

gesture_names = ['ok', 'ok_left', 'palm', 'palm_left']

model = load_model('gesture_detector.model')

def predict_image(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)

    # model.predict() returns an array of possibilities
    result = gesture_names[np.argmax(pred_array)]

    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()

    cv2.imshow('webcam', frame)

    k = cv2.waitKey(10)
    if k == 32:
        # frame = np.stack((frame,)*3, axis=-1)     # breaks the app
        frame = cv2.resize(frame, (28, 28))
        frame = frame.reshape(1, 28, 28, 3)
        prediction, score = predict_image(frame)