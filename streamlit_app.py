import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
model = torch.load("models/final_model.pth", map_location="cpu")
model.eval()

# ---------------------------------------------------------
# Torchaudio MEL extractor (replaces librosa)
# ---------------------------------------------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

def extract_mel(audio):
    audio_tensor = torch.tensor(audio).float()

    # reshape (samples,) → (1, samples)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    mel = mel_transform(audio_tensor)

    mel_db = torchaudio.functional.amplitude_to_DB(
        mel,
        multiplier=10,
        amin=1e-10,
        db_multiplier=0
    )

    # (1, 1, mel_bins, time)
    mel_db = mel_db.unsqueeze(0)
    return mel_db


# ---------------------------------------------------------
# Predict Emotion
# ---------------------------------------------------------
EMOTIONS = ["Angry", "Happy", "Sad", "Fear", "Neutral", "Disgust", "Surprise"]

def predict_emotion(audio):
    mel = extract_mel(audio)
    with torch.no_grad():
        out = model(mel)
        idx = torch.argmax(out).item()
    return EMOTIONS[idx]


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Speech Emotion Recognition (Upload Version)")
st.write("Upload a WAV audio file to detect emotion and get a song recommendation.")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded:
    st.audio(uploaded)

    # -------------------------------------------
    # Read audio using soundfile
    # -------------------------------------------
    audio, sr = sf.read(uploaded)
    audio = audio.astype(np.float32)

    # Convert stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if not 16k
    if sr != 16000:
        audio_tensor = torch.tensor(audio).float()
        audio = torchaudio.functional.resample(audio_tensor, sr, 16000).numpy()
        sr = 16000

    emotion = predict_emotion(audio)

    st.success(f"Detected Emotion: **{emotion}**")

    # Song Mapping
    SONG_MAP = {
        "Happy": "Happy – Pharrell Williams",
        "Sad": "Let Her Go – Passenger",
        "Angry": "Radioactive – Imagine Dragons",
        "Neutral": "Photograph – Ed Sheeran",
        "Fear": "Demons – Imagine Dragons",
        "Disgust": "Believer – Imagine Dragons",
        "Surprise": "Adventure of a Lifetime – Coldplay"
    }

    st.info(f"Recommended Song: **{SONG_MAP.get(emotion, 'No song available')}**")
