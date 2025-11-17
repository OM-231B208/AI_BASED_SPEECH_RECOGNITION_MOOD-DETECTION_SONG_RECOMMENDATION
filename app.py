
# ----------------------------------------------------
# app.py (FIXED VERSION - Marquee & Clickable Songs)
# ----------------------------------------------------

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
import os
from io import BytesIO
import plotly.graph_objects as go
from scipy.io import wavfile
import tempfile

# ----------------------------------------------------
# Streamlit page config
# ----------------------------------------------------
st.set_page_config(
    page_title="🎵 Emotion Music Recommender",
    layout="wide",
    page_icon="🎵"
)

# ----------------------------------------------------
# STYLING + FIXED HORIZONTAL MARQUEE
# ----------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    font-family: "Segoe UI", Arial, sans-serif;
}

/* FIXED horizontal marquee */
.top-marquee {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    border-radius: 10px;
    background: linear-gradient(90deg,#ff6b6b,#f7a072);
    padding: 12px;
    margin-bottom: 20px;
}
.top-marquee h2 {
    display: inline-block;
    animation: marquee 15s linear infinite;
    font-weight: bold;
    margin: 0;
    padding-left: 100%;
    font-size: 22px;
    white-space: nowrap;
}

@keyframes marquee {
    from { transform: translateX(0); }
    to { transform: translateX(-100%); }
}

.card {
    background: rgba(255,255,255,0.15);
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.20);
}

.song-card {
    background: linear-gradient(135deg,#ffafbd 0%,#ffc3a0 100%);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
    color: #2b2b2b;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s;
    cursor: pointer;
    display: inline-block;
    min-width: 200px;
    text-align: center;
}
.song-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.song-card a {
    text-decoration: none;
    color: #2b2b2b;
    display: block;
}

.song-card a:hover {
    color: #000;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-marquee">
    <h2>🎵 WELCOME TO THE WORLD OF YOUR MUSIC — Record or Upload Your Voice, Detect Emotion & Get Matched Songs 🎧</h2>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SONG DATABASE
# ----------------------------------------------------
SONGS = {
    "happy": [
        {"title": "Happy", "artist": "Pharrell Williams", "link": "https://youtu.be/ZbZSe6N_BXs"},
        {"title": "Don't Stop Me Now", "artist": "Queen", "link": "https://youtu.be/HgzGwKwLmgM"},
        {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "link": "https://youtu.be/iPUmE-tne5U"}
    ],
    "disgust": [
    {
        "title": "Toxic",
        "artist": "Britney Spears",
        "link": "https://www.youtube.com/watch?v=LOZuxwVk7TU"
    },
    {
        "title": "Take a Bow",
        "artist": "Rihanna",
        "link": "https://www.youtube.com/watch?v=J3UjJ4wKLkg"
    },
    {
        "title": "Bad Guy",
        "artist": "Billie Eilish",
        "link": "https://www.youtube.com/watch?v=DyDfgMOUjCI"
    }
    ],
    "sad": [
        {"title": "Fix You", "artist": "Coldplay", "link": "https://youtu.be/k4V3Mo61fJM"},
        {"title": "Someone Like You", "artist": "Adele", "link": "https://youtu.be/hLQl3WQQoQ0"},
        {"title": "The Night We Met", "artist": "Lord Huron", "link": "https://youtu.be/KtlgYxa6BMU"}
    ],
    "angry": [
        {"title": "Killing in the Name", "artist": "RATM", "link": "https://youtu.be/bWXazVhlyxQ"},
        {"title": "Break Stuff", "artist": "Limp Bizkit", "link": "https://youtu.be/ZpUYjpKg9KY"},
        {"title": "Bodies", "artist": "Drowning Pool", "link": "https://youtu.be/04F4xlWSFh0"}
    ],
    "neutral": [
        {"title": "Weightless", "artist": "Marconi Union", "link": "https://youtu.be/UfcAVejslrU"},
        {"title": "Clair de Lune", "artist": "Debussy", "link": "https://youtu.be/CvFH_6DNRCY"},
        {"title": "Ambient 1", "artist": "Brian Eno", "link": "https://youtu.be/vNwYtllyt3Q"}
    ],
    "fear": [
        {"title": "Mad World", "artist": "Gary Jules", "link": "https://youtu.be/4N3N1MlvVc4"},
        {"title": "Hurt", "artist": "Johnny Cash", "link": "https://youtu.be/8AHCfZTRGiI"},
        {"title": "The Sound of Silence", "artist": "Disturbed", "link": "https://youtu.be/u9Dg-g7t2l4"}
    ]
}

EMO_COLORS = {
    "happy": "#FFD93D",
    "sad": "#6C5CE7",
    "angry": "#FF6B6B",
    "neutral": "#C7CEEA",
    "fear": "#A8E6CF"
}

EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐", "fear": "😨"
}

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("emotion_model.pkl", "rb"))
        le = pickle.load(open("label_encoder.pkl", "rb"))
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, le = load_model()

# ----------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------
def extract(audio_data):
    try:
        # Handle different input types
        if isinstance(audio_data, bytes):
            audio, sr = sf.read(BytesIO(audio_data))
        else:
            audio, sr = audio_data, 22050
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Resample if needed
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            sr = 22050
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# ----------------------------------------------------
# PREDICT
# ----------------------------------------------------
def predict(feat):
    try:
        feat = feat.reshape(1, -1)
        pred = model.predict(feat)[0]
        probs = model.predict_proba(feat)[0] * 100
        return le.classes_[pred], float(probs[pred]), probs
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None, None, None

# ----------------------------------------------------
# Initialize session state
# ----------------------------------------------------
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

# ----------------------------------------------------
# LAYOUT
# ----------------------------------------------------
left, right = st.columns([1, 1.2])

# ----------------------------------------------------
# LEFT SIDE — AUDIO INPUT
# ----------------------------------------------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎤 Record or Upload Audio")

    # Audio recorder using st.audio_input (Streamlit native - simpler and more reliable)
    st.write("#### Use Browser Audio Recorder")
    audio_value = st.audio_input("Record your voice")
    
    if audio_value is not None:
        st.session_state.recorded_audio = audio_value.read()
        st.success("✅ Audio recorded successfully!")
        st.audio(st.session_state.recorded_audio, format='audio/wav')

    st.write("---")
    st.subheader("📁 Or Upload a WAV File")
    uploaded = st.file_uploader("Upload WAV", type=["wav", "mp3", "m4a", "ogg"])
    
    if uploaded is not None:
        st.audio(uploaded, format=f'audio/{uploaded.type.split("/")[1]}')
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# SELECT AUDIO SOURCE
# ----------------------------------------------------
audio_bytes = None
audio_source = None

if uploaded is not None:
    audio_bytes = uploaded.read()
    audio_source = "uploaded"
elif st.session_state.recorded_audio is not None:
    audio_bytes = st.session_state.recorded_audio
    audio_source = "recorded"

# ----------------------------------------------------
# RIGHT SIDE — OUTPUT
# ----------------------------------------------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Emotion Detection")
    
    if audio_bytes is None:
        st.info("👈 Please record or upload audio to begin analysis")
    else:
        st.success(f"Audio ready for analysis ({audio_source})")
    
    result_box = st.empty()
    gauge_box = st.empty()
    prob_box = st.empty()
    song_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# ANALYZE BUTTON
# ----------------------------------------------------
if audio_bytes:
    if st.button("🔎 Analyze Emotion", type="primary", use_container_width=True):
        if model is None or le is None:
            st.error("Model not loaded. Please check model files.")
        else:
            with st.spinner("Analyzing emotion..."):
                try:
                    # Extract features
                    feat = extract(audio_bytes)
                    
                    if feat is None:
                        st.error("Failed to extract audio features.")
                    else:
                        # Predict emotion
                        label, conf, probs = predict(feat)
                        
                        if label is None:
                            st.error("Failed to predict emotion.")
                        else:
                            emoji = EMOJI.get(label, "🎧")
                            color = EMO_COLORS.get(label, "#fff")

                            result_box.markdown(
                                f"<h2 style='text-align:center;'>{emoji} {label.upper()}</h2>"
                                f"<p style='text-align:center; font-size:18px;'>Confidence: {conf:.1f}%</p>",
                                unsafe_allow_html=True
                            )

                            # Confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=conf,
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': color},
                                    'threshold': {
                                        'line': {'color': "white", 'width': 4},
                                        'thickness': 0.75,
                                        'value': conf
                                    }
                                },
                                title={'text': "Confidence Level", 'font': {'size': 20}},
                                number={'font': {'size': 40}}
                            ))
                            fig.update_layout(
                                height=250,
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'color': 'white'}
                            )
                            gauge_box.plotly_chart(fig, use_container_width=True)

                            # Probability bar graph
                            emo_names = le.classes_
                            fig2 = go.Figure(go.Bar(
                                x=probs,
                                y=emo_names,
                                orientation='h',
                                marker={'color': [EMO_COLORS.get(e, "#ccc") for e in emo_names]},
                                text=[f'{p:.1f}%' for p in probs],
                                textposition='auto'
                            ))
                            fig2.update_layout(
                                height=300,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font={'color': 'white'},
                                xaxis={'title': 'Probability (%)', 'color': 'white'},
                                yaxis={'title': 'Emotions', 'color': 'white'},
                                title={'text': 'Emotion Probability Distribution', 'font': {'size': 16}}
                            )
                            prob_box.plotly_chart(fig2, use_container_width=True)

                            # Song recommendations with proper clickable links
                            if label in SONGS:
                                song_box.markdown("### 🎵 Recommended Songs", unsafe_allow_html=True)
                                
                                # Create columns for better layout
                                cols = st.columns(3)
                                for idx, song in enumerate(SONGS[label]):
                                    with cols[idx % 3]:
                                        st.markdown(f"""
                                        <div class="song-card">
                                            <a href="{song['link']}" target="_blank">
                                                <strong style="font-size: 16px;">{song['title']}</strong><br>
                                                <small style="font-size: 12px;">{song['artist']}</small>
                                            </a>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                song_box.info("No recommended songs for this emotion.")
                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.exception(e)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.markdown("<center>Made with ❤ using Streamlit | Emotion Detection AI</center>", unsafe_allow_html=True)

