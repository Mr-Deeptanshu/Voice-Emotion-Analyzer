Voice Emotion Analyzer 🎤

A deep learning-based system designed to detect and classify human emotions from speech audio. This project leverages advanced audio signal processing techniques and Convolutional Neural Networks (CNN) to analyze vocal patterns and predict emotions such as happy, sad, angry, and neutral.

🔍 Project Overview

Human speech carries rich emotional information beyond words. This project focuses on extracting and interpreting these emotional cues using machine learning. The system processes raw audio input, converts it into meaningful numerical features (MFCCs), and feeds them into a trained CNN model for accurate emotion classification.

The application is designed to simulate real-time emotion detection, making it suitable for use cases such as human-computer interaction, mental health analysis, and intelligent virtual assistants.
✨ Key Features
Real-time speech emotion prediction
Efficient audio preprocessing and noise handling
MFCC (Mel Frequency Cepstral Coefficients) feature extraction
Deep learning-based classification using CNN
Interactive and user-friendly frontend interface
Modular and scalable architecture

🛠 Tech Stack

Programming & Libraries:
Python
TensorFlow / Keras
Librosa
NumPy

Frontend:
HTML
CSS
JavaScript

⚙️ System Architecture
The system follows a structured pipeline:

Audio Input – Captures or loads speech audio
Preprocessing – Noise reduction and normalization
Feature Extraction – Conversion of audio into MFCC features
Model Prediction – CNN model processes features and predicts emotion
Output Display – Predicted emotion shown via frontend interface

📂 Project Structure
speech_emotion_model.h5 → Pre-trained CNN model
live_prediction.ipynb → Jupyter notebook for testing and predictions
final_check.py → Main execution script for running the system
templates/index.html → Frontend interface
css/, js/ → Styling and client-side logic

📊 Dataset
The model is trained using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains professionally recorded emotional speech samples.

Note: The dataset is not included in this repository due to its large size.

🚀 How to Run
Install all required dependencies
Run live_prediction.ipynb for testing
or
Run final_check.py for full execution
Launch the frontend interface via index.html
Provide audio input to get emotion predictions

📈 Results & Performance
The model demonstrates reliable performance in classifying emotions from speech audio. Accuracy depends on audio quality and noise levels, but preprocessing and feature extraction significantly improve prediction consistency.

🔮 Future Scope
Integration with real-time microphone input
Deployment as a web application or API
Use of advanced architectures (LSTM, Transformer models)
Multi-language emotion detection
Integration with chatbot or virtual assistant systems

👤 Author
Deeptanshu Yadav

License
This project is for educational purpose
