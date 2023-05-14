

import cv2
import numpy as np
import tensorflow as tf
import serial

model = tf.keras.models.load_model('/Users/parthkhurana/Documents/templates/pantherHack/garbage_classifier.h5')

labels = ['cardboard', 'metal', 'paper', 'plastic', 'trash']

arduino = serial.Serial('/dev/cu.usbserial-14220', 9600)
color_ranges = {
    'cardboard': [(0, 50, 50), (30, 255, 255)],
    'metal': [(80, 50, 50), (120, 255, 255)],
    'paper': [(0, 0, 150), (179, 70, 255)],
    'plastic': [(20, 50, 50), (40, 255, 255)],
    'trash': [(0, 0, 0), (179, 50, 50)]
}


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for waste_type, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # finding the number of pixels in the mask
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > 100:
            resized_mask = cv2.resize(mask, (256, 256))
            input_image = cv2.merge((resized_mask, resized_mask, resized_mask))
            input_image = input_image.astype('float32') / 255.0
            input_image = np.expand_dims(input_image, axis=0)

            prediction = model.predict(input_image)
            predicted_type = labels[np.argmax(prediction)]
            accuracy = prediction[0][np.argmax(prediction)] * 100

            # Send the predicted waste type to the Arduino board
            arduino.write((predicted_type + '\n').encode())
            arduino.flush()

            cv2.putText(frame, f'Predicted class: {predicted_type}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
    # frame display
    cv2.imshow('frame', frame)


#turn on live video camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        process_frame(frame)
        
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
