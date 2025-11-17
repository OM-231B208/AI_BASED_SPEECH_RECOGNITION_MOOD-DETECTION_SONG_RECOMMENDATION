from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
import soundfile as sf
import numpy as np
import uvicorn

app = FastAPI()

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = torch.load("models/final_model.pth", map_location="cpu")
model.eval()

# ---------------------------------------------------------
# Torchaudio MEL extractor
# ---------------------------------------------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

def extract_mel(audio):
    audio_tensor = torch.tensor(audio).float()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    mel = mel_transform(audio_tensor)

    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10, amin=1e-10, db_multiplier=0
    )

    mel_db = mel_db.unsqueeze(0)
    return mel_db

EMOTIONS = ["Angry", "Happy", "Sad", "Fear", "Neutral", "Disgust", "Surprise"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio, sr = sf.read(file.file)
    audio = audio.astype(np.float32)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio_tensor = torch.tensor(audio).float()
        audio = torchaudio.functional.resample(audio_tensor, sr, 16000).numpy()

    mel = extract_mel(audio)

    with torch.no_grad():
        out = model(mel)
        emotion_idx = torch.argmax(out).item()

    emotion = EMOTIONS[emotion_idx]

    SONG_MAP = {
        "Happy": "Happy – Pharrell Williams",
        "Sad": "Let Her Go – Passenger",
        "Angry": "Radioactive – Imagine Dragons",
        "Neutral": "Photograph – Ed Sheeran",
        "Fear": "Demons – Imagine Dragons",
        "Disgust": "Believer – Imagine Dragons",
        "Surprise": "Adventure of a Lifetime – Coldplay"
    }

    return {
        "emotion": emotion,
        "song": SONG_MAP.get(emotion, "No song available")
    }

if __name__ == "__main__":
    uvicorn.run("render_app:app", host="0.0.0.0", port=10000)
