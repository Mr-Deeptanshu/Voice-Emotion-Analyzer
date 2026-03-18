import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("artifacts/speech_emotion_model.h5")
label_classes = np.load("artifacts/label_classes.npy")

# Feature extraction function
def extract_features(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta.T, axis=0)

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2.T, axis=0)

    return np.hstack([mfcc_mean, delta_mean, delta2_mean])


# Website UI
st.title("🎤 Speech Emotion Analyzer")
st.write("Upload a WAV audio file and detect emotion")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    features = extract_features(uploaded_file)
    features = features.reshape(1, 120, 1)

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = label_classes[predicted_index]

    st.success(f"Predicted Emotion: **{predicted_emotion}**")