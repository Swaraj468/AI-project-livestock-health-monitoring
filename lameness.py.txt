import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained model
model = tf.keras.models.load_model('lameness_detection_model.h5')

# Set up video capture device
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Capture frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints using Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    # Draw keypoints on frame
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Extract features from keypoints
    features = []
    for corner in corners:
        x, y = corner.ravel()
        patch = gray[y-15:y+15, x-15:x+15]
        patch = cv2.resize(patch, (32, 32))
        patch = img_to_array(patch)
        patch = np.expand_dims(patch, axis=0)
        features.append(patch)
    features = np.vstack(features)

    # Make prediction using model
    pred = model.predict(features)

    # Get predicted class label
    if np.argmax(pred) == 0:
        label = 'healthy'
    else:
        label = 'lame'

    # Display label on frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Lameness Detection System', frame)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
