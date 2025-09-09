# -*- coding: utf-8 -*-
"""
Interactive FER2013 Emotion Prediction Script
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Suppress TensorFlow logs: 0=all logs, 1=info, 2=warnings, 3=errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Optional: suppress AVX/FMA CPU instructions messages
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf


# ----------------------------
# Load the saved model
# ----------------------------

MODEL_PATH = "face_it_api/models/DeepS_EM_09sept.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.\n")

# ----------------------------
# Emotion labels mapping
# ----------------------------
emotion_labels = {
    0: "Angry 😠",
    1: "Disgust 🤢",
    2: "Fear 😨",
    3: "Happy 😄",
    4: "Sad 😢",
    5: "Surprise 😲",
    6: "Neutral 😐"
}

# ----------------------------
# Prediction function
# ----------------------------
def predict_emotion(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.\n")
        return

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'.\n")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face using Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

    if len(faces) == 0:
        print("No face detected, using full image for prediction (less accurate).")
        face_crop = cv2.resize(gray, (48, 48))
    else:
        x, y, w, h = faces[0]  # take first detected face
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48, 48))

    # Preprocess for model
    input_face = np.expand_dims(np.expand_dims(face_crop, axis=-1), axis=0) / 255.0

    # Predict emotion
    pred_probs = model.predict(input_face, verbose=0)
    pred_class = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100

    # Display image with prediction
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {emotion_labels[pred_class]} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    # Print results
    print("🎯 Prediction Results 🎯")
    print(f"😊 Predicted Emotion : {emotion_labels[pred_class]}")
    print(f"📊 Confidence Level : {confidence:.2f}%\n")

# ----------------------------
# Interactive loop
# ----------------------------
if __name__ == "__main__":
    print("😠🤢😨😄😢 DeepS Emotion Predictor 😢😄😨🤢😠")
    print("Type 'exit' to quit.\n")
    while True:
        image_path = input("Enter image path: ").strip()
        if image_path.lower() == 'exit':
            print("Exiting...")
            break
        predict_emotion(image_path)
