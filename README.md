# AI_BASED_SPEECH_RECOGNITION_MOOD-DETECTION_SONG_RECOMMENDATION
# Speech Emotion Recognition + Mood-Based Song Recommendation

This project builds a real-time Speech Emotion Recognition (SER) system that listens to the user's voice, detects their emotional state using a deep learning model, and recommends a song based on the detected emotion. The system uses audio preprocessing (MFCCs / Mel-Spectrogram), a trained CRNN/ResNet-based model, and a FastAPI web interface.

---

## 🎯 Features
- Real-time microphone audio recording
- Noise reduction + stereo-to-mono preprocessing
- Deep learning–based emotion classification
- Supported emotions:
  - Angry
  - Happy
  - Sad
  - Fear
  - Neutral
  - Disgust
  - Surprise
- Automatic song recommendation for each emotion
- FastAPI backend for prediction
- Web UI for demo / deployment
- Fully deployable on Render / HuggingFace / GitHub

---

## 🏗️ Project Structure


---

## 🔧 How It Works
1. Captures audio using `sounddevice` or browser recorder.
2. Converts stereo → mono (for safe preprocessing).
3. Applies Mel-Spectrogram or MFCC feature extraction.
4. Loads trained model (TensorFlow/PyTorch).
5. Predicts emotion.
6. Recommends song mapped from the emotion.
7. Displays result on the UI.

---

## 🚀 Running the Project Locally

### 1. Install dependencies

