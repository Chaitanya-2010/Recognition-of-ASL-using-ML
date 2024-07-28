import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import pyttsx3

# engine = pyttsx3.init()

top, right, bottom, left = 100, 150, 400, 450
# load saved model from PC
model = tf.keras.models.load_model(r'C:\Users\Rajesh Vishwakarma\Desktop\project\model\ResNet50_ASL_4.h5')
model.summary()

data_dir = r'C:\Users\Rajesh Vishwakarma\Desktop\project\random3\train'
# getting the labels form data directory
labels = sorted(os.listdir(data_dir))
print(labels)

# initiating the video source, 0 for internal camera
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
cap = cv2.VideoCapture(2)

while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # region of intrest
    roi = frame[top:bottom, right:left]
    roi = cv2.GaussianBlur(roi, (7, 7), 0)

    img = cv2.resize(roi, (96, 96))

    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    img = cv2.flip(img, 1)

    img = img

    prediction = model.predict(img.reshape(1, 96, 96, 3))
    char_index = np.argmax(prediction)
    # print(char_index)

    confidence = round(prediction[0, char_index] * 100, 1)
    predicted_char = labels[char_index]
    # print(predicted_char)
    # engine = pyttsx3.init()
    # if predicted_char != "nothing":
    #     engine.say(predicted_char)
    #     engine.runAndWait()

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL = 5
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    # writing the predicted char and its confidence percentage to the frame
    msg = predicted_char + ', Conf: ' + str(confidence) + ' %'
    cv2.putText(roi, msg, (10, 20), font, fontScale, color, thickness)

    cv2.imshow('roi', roi)
    cv2.imshow('frame', frame)

    # close the camera when press 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
