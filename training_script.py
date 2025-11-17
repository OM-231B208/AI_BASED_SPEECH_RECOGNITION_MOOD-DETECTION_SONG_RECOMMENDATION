import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import warnings
warnings.filterwarnings('ignore')

# ==================== STEP 1: LOAD DATA ====================
print("="*70)
print("LOADING DATASET...".center(70))
print("="*70)

DATASET_PATH = r'C:\Users\om885\OneDrive\Desktop\new_ai_model\Deep-Learning-Projects\Speech Emotion Recognition - Sound Classification\TESS Toronto emotional speech set data'

paths = []
labels = []

for dirname, _, filenames in os.walk(DATASET_PATH):
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1].split('.')[0].lower()
            labels.append(label)

df = pd.DataFrame({'speech': paths, 'label': labels})
print(f"✅ Loaded {len(df)} audio files")
print(f"✅ Emotions found: {df['label'].unique()}")
print("\n" + str(df['label'].value_counts()))

# ==================== STEP 2: EXTRACT FEATURES (SIMPLE) ====================
print("\n" + "="*70)
print("EXTRACTING FEATURES...".center(70))
print("="*70)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=2.5, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except:
        return None

features = []
valid_labels = []

for i, row in df.iterrows():
    feat = extract_features(row['speech'])
    if feat is not None:
        features.append(feat)
        valid_labels.append(row['label'])
    if (i+1) % 200 == 0:
        print(f"Processed {i+1}/{len(df)} files...")

X = np.array(features)
y = np.array(valid_labels)

print(f"\n✅ Features extracted: {X.shape}")

# ==================== STEP 3: ENCODE LABELS ====================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n✅ Emotion Classes: {le.classes_}")

# ==================== STEP 4: SPLIT DATA ====================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# ==================== STEP 5: TRAIN MODEL (SIMPLE & FAST) ====================
print("\n" + "="*70)
print("TRAINING MODEL...".center(70))
print("="*70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Simple Random Forest - Fast and Reliable
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "╔" + "═"*68 + "╗")
print("║" + " "*68 + "║")
print("║" + f"🎯 MODEL ACCURACY: {accuracy*100:.2f}%".center(68) + "║")
print("║" + " "*68 + "║")
print("╚" + "═"*68 + "╝")

# ==================== STEP 6: VISUALIZE RESULTS ====================
print("\n" + "="*70)
print("GENERATING PERFORMANCE REPORT...".center(70))
print("="*70)

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Accuracy by Class
class_acc = []
for i in range(len(le.classes_)):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).sum() / mask.sum() * 100
        class_acc.append(acc)

axes[1].barh(le.classes_, class_acc, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Per-Emotion Accuracy', fontsize=16, fontweight='bold')
axes[1].set_xlim(0, 110)

for i, v in enumerate(class_acc):
    axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.show()
# ==================== SAVE MODEL FOR WEB APP ====================
import pickle

print("\n" + "="*70)
print("SAVING MODEL FOR WEB APP...".center(70))
print("="*70)

# Save the trained model
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved as 'emotion_model.pkl'")

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Label encoder saved as 'label_encoder.pkl'")

print("\n🎉 Files created in current directory:")
print("   📁 emotion_model.pkl")
print("   📁 label_encoder.pkl")
print("\n✅ Ready for web app deployment!")
print("="*70)
# Classification Report
print("\n📋 CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==================== STEP 7: SONG RECOMMENDATIONS ====================
print("\n" + "="*70)
print("SONG RECOMMENDATION SYSTEM".center(70))
print("="*70)

SONGS = {
    'angry': [
        "Break Stuff - Limp Bizkit",
        "Killing in the Name - Rage Against the Machine",
        "Bodies - Drowning Pool"
    ],
    'disgust': [
        "Toxic - Britney Spears",
        "Bad Guy - Billie Eilish",
        "Tainted Love - Soft Cell"
    ],
    'fear': [
        "Mad World - Gary Jules",
        "Hurt - Johnny Cash",
        "Creep - Radiohead"
    ],
    'happy': [
        "Happy - Pharrell Williams",
        "Walking on Sunshine - Katrina and the Waves",
        "Good Vibrations - The Beach Boys"
    ],
    'neutral': [
        "Weightless - Marconi Union",
        "Clair de Lune - Debussy",
        "Porcelain - Moby"
    ],
    'ps': [
        "Lovely Day - Bill Withers",
        "Mr. Blue Sky - ELO",
        "Here Comes the Sun - The Beatles"
    ],
    'sad': [
        "Someone Like You - Adele",
        "The Night We Met - Lord Huron",
        "Fix You - Coldplay"
    ]
}

def recommend_songs(emotion):
    emotion = emotion.lower()
    if emotion in SONGS:
        return SONGS[emotion]
    return ["No recommendations available"]

def predict_and_recommend(audio_path):
    # Extract features
    features = extract_features(audio_path)
    if features is None:
        print("❌ Error processing audio file")
        return
    
    features = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    emotion = le.classes_[prediction]
    
    # Get probabilities
    proba = model.predict_proba(features)[0]
    confidence = proba[prediction] * 100
    
    # Display results
    print("\n" + "="*70)
    print("🎯 EMOTION DETECTION RESULT".center(70))
    print("="*70)
    print(f"Detected Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nTop 3 Predictions:")
    top_3 = np.argsort(proba)[-3:][::-1]
    for idx in top_3:
        print(f"  • {le.classes_[idx]}: {proba[idx]*100:.2f}%")
    
    # Recommend songs
    songs = recommend_songs(emotion)
    print("\n" + "="*70)
    print(f"🎵 RECOMMENDED SONGS FOR '{emotion.upper()}'".center(70))
    print("="*70)
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song}")
    print("="*70)
    
    return emotion, confidence

# ==================== STEP 8: TEST THE SYSTEM ====================
print("\n" + "="*70)
print("TESTING THE SYSTEM".center(70))
print("="*70)

# Test with a random file
test_file = df['speech'].iloc[0]
print(f"\nTesting with: {os.path.basename(test_file)}")

emotion, conf = predict_and_recommend(test_file)

# ==================== STEP 9: BATCH TEST ====================
print("\n" + "="*70)
print("BATCH TESTING (5 SAMPLES)".center(70))
print("="*70)

results = []
for i in range(5):
    test_file = df['speech'].iloc[i]
    print(f"\n[Sample {i+1}] {os.path.basename(test_file)}")
    emotion, conf = predict_and_recommend(test_file)
    results.append({'file': os.path.basename(test_file), 'emotion': emotion, 'confidence': conf})

# Summary
print("\n" + "="*70)
print("📊 BATCH TEST SUMMARY".center(70))
print("="*70)
for i, res in enumerate(results, 1):
    print(f"{i}. {res['emotion'].upper()} ({res['confidence']:.1f}%) - {res['file']}")

print("\n" + "="*70)
print("✅ SYSTEM READY FOR DEMONSTRATION!".center(70))
print("="*70)

print(f"\n📌 FINAL ACCURACY: {accuracy*100:.2f}%")
print("📌 MODEL: Random Forest (Fast & Reliable)")
print("\n" + "="*70)