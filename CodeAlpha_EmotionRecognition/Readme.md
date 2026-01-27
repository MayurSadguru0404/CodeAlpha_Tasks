🎙️ Speech Emotion Recognition using Deep Learning
📌 Project Overview

This project focuses on recognizing human emotions from speech audio using deep learning and speech signal processing techniques.
The system analyzes voice signals and predicts emotions such as Happy, Sad, Angry, Neutral, Fearful, Disgust, and Surprised.

This project was developed as part of a Machine Learning Internship, with an emphasis on practical implementation rather than just theory.

🎯 Objective

To build a machine learning system that can:

Extract meaningful features from speech signals

Learn emotional patterns from audio data

Predict the emotion expressed in human speech

🧠 Approach

Speech Signal Processing

Extracted MFCC (Mel-Frequency Cepstral Coefficients) and their temporal variations

Deep Learning

Used a Bidirectional LSTM (BiLSTM) model to capture time-based dependencies in audio

Model Training & Evaluation

Trained and evaluated the model on a real-world emotional speech dataset

🗂️ Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Professionally recorded emotional speech samples

Multiple emotions spoken by different actors

Widely used benchmark dataset in speech emotion recognition

📁 Dataset Structure:

datasets/
└── RAVDESS/
    ├── Actor_01/
    ├── Actor_02/
    └── ...

⚙️ Technologies Used

Python 3.10

Librosa – audio processing & MFCC extraction

NumPy & Scikit-learn – data handling and preprocessing

TensorFlow / Keras – deep learning model

Matplotlib – training & accuracy visualization

🏗️ Model Architecture

MFCC + Delta + Delta-Delta features

Bidirectional LSTM layers

Dropout & Batch Normalization for regularization

Softmax output layer for multi-class emotion classification

📊 Results
Training Accuracy: ~80%
Test Accuracy: ~74%

Note: Speech emotion recognition is a challenging task due to speaker variations, recording conditions, and emotional overlap.
The achieved accuracy is considered good for internship-level and academic projects.

🔍 Sample Prediction Output
Predicted Emotion: happy
Confidence: 0.82


📈 Future Improvements

Train on multiple datasets (TESS, EMO-DB)

Use CNN + LSTM hybrid models

Apply advanced data augmentation techniques
