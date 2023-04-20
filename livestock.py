# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained model
model = tf.keras.models.load_model('livestock_model.h5')

# Set up video capture device
cap = cv2.VideoCapture(0)

# Define labels for classes
classes = ['healthy', 'sick']

# Loop through video frames
while True:
    # Capture frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize and preprocess image
    img = cv2.resize(gray, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make prediction using model
    pred = model.predict(img)

    # Get predicted class label
    label = classes[np.argmax(pred)]

    # Display label on frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Livestock Health Monitoring System', frame)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
