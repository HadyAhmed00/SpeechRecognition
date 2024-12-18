import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr
# import PyAudio
import ffmpeg
from pydub import AudioSegment


# Function to extract DSP-based features
def extract_dsp_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load FLAC file
    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)

    # Combine features into a single vector
    features = np.hstack((mfcc, zcr, spectral_centroid, spectral_bandwidth))
    return features


# Function to prepare training data
def prepare_training_data(data_path):
    features, labels = [], []
    for speaker in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                if file_name.endswith(".flac"):  # Process only FLAC files
                    file_path = os.path.join(speaker_path, file_name)
                    feature = extract_dsp_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(speaker)
    return np.array(features), np.array(labels)


# Function to train the speaker recognition model
def train_model(features, labels):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Training accuracy: {accuracy * 100:.2f}%")
    return model, encoder


# Function to visualize audio signal
def visualize_signal(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform of Audio Signal ({file_path})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# Function to convert WAV to FLAC
def convert_wav_to_flac(wav_path, flac_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(flac_path, format="flac")


# Function to recognize speaker
import os
import speech_recognition as sr
from pydub import AudioSegment

# Configure pydub to use FFmpeg explicitly
AudioSegment.converter = r"C:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  # Update with your FFmpeg path


def recognize_speaker(model, encoder, recognizer, temp_audio_path="temp.wav", flac_audio_path="temp.flac", timeout=10):
    """
    Function to capture audio from the microphone and recognize the speaker.
    """
    print("Listening...")
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
            print("Speak now:")
            audio = recognizer.listen(source, timeout=timeout)  # Listen for the audio

            # Save microphone audio to WAV format
            with open(temp_audio_path, "wb") as f:
                f.write(audio.get_wav_data())

            # Convert WAV to FLAC
            audio_segment = AudioSegment.from_wav(temp_audio_path)
            audio_segment.export(flac_audio_path, format="flac")
            print(f"Audio saved and converted to FLAC: {flac_audio_path}")

    except sr.WaitTimeoutError:
        print("Microphone listening timed out. No input detected.")
        return
    except Exception as e:
        print(f"Error while capturing audio: {e}")
        return

    # Extract DSP features from the FLAC file
    try:
        # visualize_signal(flac_audio_path)
        feature = extract_dsp_features(flac_audio_path)
        feature = feature.reshape(1, -1)

        # Get prediction probabilities
        probabilities = model.predict_proba(feature)[0]
        max_prob = max(probabilities)
        predicted_label = model.predict(feature)[0]
        speaker = encoder.inverse_transform([predicted_label])[0]

        # Check confidence
        if max_prob < 0.1:  # Less than 10% confidence
            print("Speaker not recognized. Confidence is too low.")
        else:
            print(f"This is {speaker}'s voice with confidence: {max_prob * 100:.2f}%")

    except Exception as e:
        print(f"Error during feature extraction or prediction: {e}")


# Main Function
if __name__ == "__main__":
    DATA_PATH = "DataSet"  # Folder containing subfolders of FLAC files
    recognizer = sr.Recognizer()

    # Step 1: Visualize example audio
    # example_audio = "DataSet/Yousef/yousef (1).flac"
    # visualize_signal(example_audio)

    # Step 2: Prepare data and train the model
    features, labels = prepare_training_data(DATA_PATH)
    model, encoder = train_model(features, labels)

    # Step 3: Recognize new voice input
    while True:
        recognize_speaker(model, encoder, recognizer)
