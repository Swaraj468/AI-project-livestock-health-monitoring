import sounddevice as sd
import numpy as np
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('respiratory_detection_model.h5')

# Set up audio stream
fs = 44100
duration = 5
channels = 1

# Loop through audio frames
while True:
    # Record audio
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=channels)
    sd.wait()

    # Preprocess audio
    recording = np.mean(recording, axis=1)
    recording = np.reshape(recording, (1, -1, 1))

    # Make prediction using model
    pred = model.predict(recording)

    # Get predicted class label
    if np.argmax(pred) == 0:
        label = 'healthy'
    else:
        label = 'respiratory problem'

    # Print label
    print('Livestock is', label)
