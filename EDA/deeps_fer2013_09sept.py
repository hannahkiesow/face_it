# -*- coding: utf-8 -*-
"""
DeepS FER2013 Emotion Recognition - Prediction & Evaluation
Uses pre-trained model saved as .keras file.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# Configuration
# ---------------------------
FER_CSV_PATH = "/home/deepa/code/hannahkiesow/face_it/EDA/data/fer2013.csv"
MODEL_PATH = "/home/deepa/code/hannahkiesow/face_it/face_it_api/models/DeepS_emotion_model.keras"
BATCH_SIZE = 64
IMAGE_SIZE = 48
NUM_CLASSES = 7

# Emotion labels
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates().dropna()

    # Convert pixels to arrays
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

    # Filter correct size
    df = df[df['pixels'].apply(lambda x: len(x) == IMAGE_SIZE*IMAGE_SIZE)]
    X = np.stack(df['pixels'].values).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0

    y = df['emotion'].values
    y_onehot = to_categorical(y, NUM_CLASSES)

    # Split by Usage
    X_train = X[df['Usage'] == "Training"]
    y_train = y_onehot[df['Usage'] == "Training"]
    X_val = X[df['Usage'] == "PublicTest"]
    y_val = y_onehot[df['Usage'] == "PublicTest"]
    X_test = X[df['Usage'] == "PrivateTest"]
    y_test = y_onehot[df['Usage'] == "PrivateTest"]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------------------------
# Evaluate Model
# ---------------------------
def evaluate_model(model, X_test, y_test):
    test_gen = ImageDataGenerator().flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(EMOTION_LABELS.values())))

# ---------------------------
# Predict Single Image
# ---------------------------
def predict_image(model, img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20,20))

    if len(faces) == 0:
        face_crop = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    else:
        x, y, w, h = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))

    input_face = np.expand_dims(face_crop, axis=(0,-1)) / 255.0
    pred_prob = model.predict(input_face)
    pred_class = np.argmax(pred_prob)
    confidence = np.max(pred_prob) * 100

    print(f"Predicted emotion: {EMOTION_LABELS[pred_class]} ({confidence:.2f}%)")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{EMOTION_LABELS[pred_class]} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    # Load model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(FER_CSV_PATH)
    print(f"Data loaded. Test set shape: {X_test.shape}, {y_test.shape}")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Example single image prediction (update path)
    # predict_image(model, "sample_face.png")

if __name__ == "__main__":
    main()
