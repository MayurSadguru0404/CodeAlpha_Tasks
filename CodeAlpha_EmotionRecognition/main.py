import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


class EmotionRecognitionSystem:
    def __init__(self, sample_rate=22050, duration=3, n_mfcc=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.max_pad_len = 174
        self.label_encoder = LabelEncoder()
        self.model = None

    def extract_features(self, audio_path, augment=False):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            if augment:
                audio = self._augment_audio(audio, sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            # Pad or truncate
            if combined.shape[1] > self.max_pad_len:
                combined = combined[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - combined.shape[1]
                combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
            return combined.T
        except Exception as e:
            print(f"Error: {audio_path} -> {e}")
            return None

    def _augment_audio(self, audio, sr):
        aug_type = np.random.randint(0, 4)
        if aug_type == 0:
            noise = np.random.randn(len(audio))
            audio = audio + 0.005 * noise
        elif aug_type == 1:
            rate = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(y=audio, rate=rate)
        elif aug_type == 2:
            n_steps = np.random.randint(-3, 3)
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
        elif aug_type == 3:
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
        return audio

    def load_dataset(self, dataset_path, augment_data=True):
        features, labels = [], []
        print("Step 1: Loading dataset and extracting features...")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    path = os.path.join(root, file)
                    try:
                        emotion_code = int(file.split("-")[2])
                        emotion_map = {1:'neutral',2:'calm',3:'happy',4:'sad',
                                       5:'angry',6:'fearful',7:'disgust',8:'surprised'}
                        emotion = emotion_map.get(emotion_code)
                        if emotion:
                            feat = self.extract_features(path)
                            if feat is not None:
                                features.append(feat)
                                labels.append(emotion)
                            if augment_data:
                                feat_aug = self.extract_features(path, augment=True)
                                if feat_aug is not None:
                                    features.append(feat_aug)
                                    labels.append(emotion)
                    except:
                        continue
        features = np.array(features)
        labels = np.array(labels)
        print(f"Loaded {len(features)} samples with {len(np.unique(labels))} emotions")
        return features, labels

    def prepare_data(self, features, labels, test_size=0.2, val_size=0.1):
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_cat = to_categorical(labels_encoded)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels_cat, test_size=test_size, random_state=42, stratify=labels_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42
        )
        print(f"Data split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=5, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        self.model = self.build_model(input_shape, num_classes)
        cb_list = [
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        ]
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size, callbacks=cb_list, verbose=1)
        return history

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        acc = accuracy_score(y_test_classes, y_pred_classes)
        print(f"Test Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test_classes, y_pred_classes, target_names=self.label_encoder.classes_))
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        return acc

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def predict_emotion(self, audio_path):
        feat = self.extract_features(audio_path)
        if feat is None:
            return None
        feat = np.expand_dims(feat, axis=0)
        pred = self.model.predict(feat)
        emotion = self.label_encoder.classes_[np.argmax(pred)]
        conf = np.max(pred)
        return emotion, conf


if __name__ == "__main__":
    dataset_path = r"C:\complete web development\Code_Alpha_Tasks\CodeAlpha_EmotionRecognition\datasets\RAVDESS"
    ers = EmotionRecognitionSystem()

    features, labels = ers.load_dataset(dataset_path, augment_data=True)
    X_train, X_val, X_test, y_train, y_val, y_test = ers.prepare_data(features, labels)

    history = ers.train_model(X_train, y_train, X_val, y_val, epochs=50)
    ers.evaluate_model(X_test, y_test)

    # Test single audio
    test_audio = r"C:\complete web development\Code_Alpha_Tasks\CodeAlpha_EmotionRecognition\datasets\RAVDESS\Actor_02\03-01-01-01-01-01-02.wav"
    emotion, conf = ers.predict_emotion(test_audio)
    print(f"\nPredicted Emotion: {emotion} with confidence {conf:.2f}")
